"""
Episode loop and batched driver.

An episode is a generator: it yields (role, messages, output_type) for each LLM call and gets the
validated output back. `simulate` keeps up to `concurrency` episodes in flight and sends all their
pending calls to the backends as one batch per round, so vLLM always works on large batches.
"""

import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Iterator, Optional, Sequence

from triagesim import prompts
from triagesim.cases import Case
from triagesim.llm import Backend, GenerationError, Request, generate
from triagesim.personas import NursePersona, PatientPersona
from triagesim.schemas import (
    VITALS,
    BeliefExtraction,
    InSitu,
    PatientOutput,
    PatientScript,
    Reading,
    Verdict,
    nurse_output_type,
)

DEFAULT_SAMPLING = {
    "nurse": {"max_tokens": 1024},
    "patient": {"max_tokens": 512},
    "belief": {"max_tokens": 512, "temperature": 0.0},
    "judge": {"max_tokens": 512, "temperature": 0.0},
    "script": {"max_tokens": 2048},
}
MAX_FLAGS_PER_LOG = 5

Call = tuple[str, list[dict[str, str]], type]
EpisodeGen = Generator[Call, Any, dict]


@dataclass
class _Episode:
    episode_id: str
    case: Case
    algorithm: str
    nurse_persona: NursePersona
    patient_persona: PatientPersona
    patient_view: str  # what the nurse sees at the triage desk


def _reviewed(
    speaker: str,
    messages: list[dict[str, str]],
    output_type: type,
    judge_messages: Optional[Callable[[str], list[dict[str, str]]]],
    max_regenerations: int,
    regenerate_as: type,
):
    """Yield a speaker call; if a judge is set, check the utterance and regenerate with its critique.

    Returns (output, verification). Verification is None when nothing was checked (no judge, or the
    nurse chose an action other than speaking); an utterance that still fails after
    `max_regenerations` is kept and flagged with passed=False, with the rejected drafts.
    """
    rejected: list[dict] = []
    current = messages
    while True:
        out = yield speaker, current, output_type
        if judge_messages is None or getattr(out, "action", "utterance") != "utterance":
            return out, None
        try:
            verdict: Verdict = yield "judge", judge_messages(out.utterance), Verdict
        except GenerationError as e:  # an unavailable judge must not end the dialogue
            return out, {"passed": None, "error": str(e), "rejected": rejected}
        if verdict.passed:
            return out, {"passed": True, "verdict": verdict.model_dump(), "rejected": rejected}
        if len(rejected) == max_regenerations:
            return out, {"passed": False, "verdict": verdict.model_dump(), "rejected": rejected}
        rejected.append({"utterance": out.utterance, "verdict": verdict.model_dump()})
        current = prompts.with_feedback(messages, out.model_dump_json(), verdict.critique)
        output_type = regenerate_as  # a regenerated nurse line must still be an utterance


def _nurse_phase(ep: _Episode, history: list, trace: list, red_flags: list, turn: int, review: bool, max_regen: int):
    """Nurse micro-steps until it speaks or ends triage; returns True if it ended.

    Within a phase the nurse may check one vital and log red flags once before speaking. Its first
    step after a patient reply records what it understood from that reply on the reply's entry.
    """
    vital_checked = flag_logged = False
    unread = history[-1] if history and history[-1]["actor"] == "patient" else None
    while True:
        reading = unread is not None
        known = {h["name"] for h in history if h.get("event") == "vital"}
        vitals = () if vital_checked else tuple(v for v in VITALS if v not in known)
        actions = (
            ("utterance",)
            + (("check_vital",) if vitals else ())
            + (() if flag_logged else ("log_red_flag",))
            + ("end",)
        )
        messages = prompts.nurse_messages(ep.nurse_persona, ep.algorithm, ep.patient_view, history, actions, vitals)

        def judge(utterance: str) -> list[dict[str, str]]:
            return prompts.judge_messages(
                "nurse", utterance, ep.nurse_persona, history, algorithm=ep.algorithm, patient=ep.patient_view
            )

        out, verification = yield from _reviewed(
            "nurse",
            messages,
            nurse_output_type(actions, vitals, reading),
            judge if review else None,
            max_regen,
            nurse_output_type(("utterance",), (), reading),
        )
        trace.append({"turn": turn, **out.model_dump()})
        if reading:
            unread["information"]["understood"] = out.understood
            unread = None

        if out.action == "check_vital":
            value = ep.case.vitals.get(out.vital)
            history.append({"turn": turn, "actor": "system", "event": "vital", "name": out.vital, "value": value})
            vital_checked = True
        elif out.action == "log_red_flag":
            seen = {" ".join(f.lower().split()) for f in red_flags}
            new = []
            for flag in out.red_flags:
                key = " ".join(flag.lower().split())
                if key and key not in seen and len(new) < MAX_FLAGS_PER_LOG:
                    seen.add(key)
                    new.append(flag.strip())
            red_flags.extend(new)
            flag_logged = True
        elif out.action == "end":
            history.append({"turn": turn, "actor": "system", "event": "triage_end", "triage": out.triage})
            return True
        else:
            entry = {"turn": turn, "actor": "nurse", "utterance": out.utterance, "triage": out.triage}
            if verification is not None:
                entry["verification"] = verification
            history.append(entry)
            return False


def _run_episode(ep: _Episode, max_turns: int, extract_beliefs: bool, review: bool, max_regen: int) -> EpisodeGen:
    history: list[dict] = []
    trace: list[dict] = []
    red_flags: list[str] = []
    beliefs: list[dict] = []
    ended, error = False, None
    setup: dict[str, Any] = {}  # the dialogue master's in-situ check and script, or why they are missing
    script: Optional[PatientScript] = None

    def artifact(status: str, **fields) -> dict:
        return {
            "episode_id": ep.episode_id,
            "status": status,
            "algorithm": ep.algorithm,
            "case": ep.case.model_dump(),
            "nurse_persona": ep.nurse_persona.model_dump(),
            "patient_persona": ep.patient_persona.model_dump(),
            **setup,
            **fields,
        }

    if ep.case.acuity is None:
        return artifact("skipped", skip_reason="no ground-truth acuity")
    if review:
        try:
            check = yield "judge", prompts.in_situ_messages(ep.case), InSitu
            setup["in_situ"] = check.model_dump()
            if not check.conversation:
                return artifact("skipped", skip_reason=check.reasoning)
        except GenerationError as e:  # simulate anyway
            setup["in_situ_error"] = str(e)
        try:
            script = yield "script", prompts.script_messages(ep.case, ep.patient_persona), PatientScript
        except GenerationError as e:  # the patient improvises from the case instead
            setup["script_error"] = str(e)
    setup["patient_script"] = None if script is None else script.model_dump()

    try:
        for turn in range(max_turns):
            ended = yield from _nurse_phase(ep, history, trace, red_flags, turn, review, max_regen)
            if ended:
                break

            def judge(utterance: str) -> list[dict[str, str]]:
                return prompts.judge_messages(
                    "patient", utterance, ep.patient_persona, history, case=ep.case, script=script
                )

            reply, verification = yield from _reviewed(
                "patient",
                prompts.patient_messages(ep.patient_persona, ep.case, script, history),
                PatientOutput,
                judge if review else None,
                max_regen,
                PatientOutput,
            )
            # what passed: the patient's own account, the master's reading of the words alone, and
            # (filled in at the nurse's next step) what the nurse understood
            info = {"disclosed": reply.disclosed, "conveyed": None, "understood": None}
            entry = {"turn": turn, "actor": "patient", "utterance": reply.utterance, "information": info}
            if verification is not None:
                entry["verification"] = verification
            history.append(entry)
            if review:
                try:
                    reading = yield "judge", prompts.reading_messages("patient", reply.utterance), Reading
                    info["conveyed"] = reading.conveyed
                except GenerationError as e:
                    info["conveyed_error"] = str(e)
            if extract_beliefs:
                nurse_said = history[-2]["utterance"]
                try:  # beliefs are auxiliary: a failed extraction must not end the dialogue
                    belief = yield "belief", prompts.belief_messages(nurse_said, reply.utterance), BeliefExtraction
                    beliefs.append({"turn": turn, **belief.model_dump()})
                except GenerationError as e:
                    beliefs.append({"turn": turn, "error": str(e)})
    except GenerationError as e:
        error = str(e)

    return artifact(
        "error" if error else "ok",
        error=error,
        end_reason="error" if error else "nurse_end" if ended else "max_turns",
        final_triage=trace[-1]["triage"] if trace else None,
        num_turns=sum(h["actor"] == "patient" for h in history),
        history=history,
        trace=trace,
        red_flags=red_flags,
        beliefs=beliefs,
    )


def _generate_groups(groups: list[tuple[Backend, list[tuple[str, Request]]]]) -> list[tuple[str, Any]]:
    def run(group):
        backend, items = group
        return list(zip([eid for eid, _ in items], generate(backend, [r for _, r in items])))

    if len(groups) == 1:
        return run(groups[0])
    with ThreadPoolExecutor(len(groups)) as pool:  # e.g. nurse on OpenRouter, patient on vLLM
        return [x for res in pool.map(run, groups) for x in res]


def simulate(
    cases: Iterable[Case],
    *,
    nurse_llm: Backend,
    patient_llm: Backend,
    nurse_personas: Sequence[NursePersona],
    patient_personas: Sequence[PatientPersona],
    algorithm: Optional[str] = None,
    belief_llm: Optional[Backend] = None,
    extract_beliefs: bool = True,
    judge_llm: Optional[Backend] = None,
    max_regenerations: int = 1,
    max_turns: int = 12,
    episodes_per_case: int = 1,
    seed: Optional[int] = 0,
    concurrency: int = 256,
    sampling: Optional[dict[str, dict]] = None,
    skip: Iterable[str] = (),
) -> Iterator[dict]:
    """Simulate triage episodes, yielding one artifact dict per episode as it finishes.

    Episode ids are "<case_id>-<k>" for k < episodes_per_case. Personas (the patient taking the
    case's gender) are drawn per episode from `seed` and the episode id, so a run is reproducible and
    ids in `skip` (e.g. already done) can be left out without changing the rest.

    `algorithm` ("esi"/"ats") applies to every case; by default each case uses its `acuity_scale`,
    else ESI. The belief extractor defaults to the nurse's backend. With a `judge_llm` (the dialogue
    master), each episode starts with the master checking that the case would have a triage
    conversation in situ (status "skipped" with its reasoning if not) and writing the patient's
    script from the case and the patient persona; every spoken line is then checked and regenerated
    with feedback up to `max_regenerations` times, then flagged. Cases without a ground-truth acuity
    are skipped. `sampling` overrides DEFAULT_SAMPLING per role ("nurse", "patient", "belief",
    "judge", "script"). Episodes whose LLM calls keep failing are yielded with status "error" and the
    dialogue up to the failure.
    """
    if algorithm is not None and algorithm not in prompts.ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {sorted(prompts.ALGORITHMS)}")
    if not nurse_personas or not patient_personas:
        raise ValueError("need at least one nurse persona and one patient persona")
    if unknown := set(sampling or {}) - set(DEFAULT_SAMPLING):
        raise ValueError(f"unknown sampling roles {sorted(unknown)}; expected {sorted(DEFAULT_SAMPLING)}")

    llms = {
        "nurse": nurse_llm,
        "patient": patient_llm,
        "belief": belief_llm or nurse_llm,
        "judge": judge_llm,
        "script": judge_llm,
    }
    role_sampling = {role: {**DEFAULT_SAMPLING[role], **(sampling or {}).get(role, {})} for role in llms}
    models = {"nurse": nurse_llm.model, "patient": patient_llm.model}
    if extract_beliefs:
        models["belief"] = llms["belief"].model
    if judge_llm is not None:
        models["judge"] = judge_llm.model
    review = judge_llm is not None
    skip = set(skip)

    def planned() -> Iterator[_Episode]:
        for case in cases:
            for k in range(episodes_per_case):
                eid = f"{case.case_id}-{k}"
                if eid in skip:
                    continue
                rng = random.Random(f"{seed}/{eid}")
                nurse, patient = rng.choice(nurse_personas), rng.choice(patient_personas)
                if case.gender:  # gender comes from the case
                    patient = patient.model_copy(update={"gender": case.gender})
                algo = algorithm or case.acuity_scale or "esi"
                yield _Episode(eid, case, algo, nurse, patient, prompts.patient_view(case, patient))

    def drive() -> Iterator[dict]:
        todo = planned()
        active: dict[str, list] = {}  # episode id -> [generator, pending call, calls made]
        finished: list[dict] = []  # episodes that ended before their first LLM call

        def fill():
            while len(active) < concurrency and (ep := next(todo, None)) is not None:
                gen = _run_episode(ep, max_turns, extract_beliefs, review, max_regenerations)
                try:
                    active[ep.episode_id] = [gen, next(gen), 0]
                except StopIteration as stop:
                    finished.append(stop.value)

        fill()
        while active or finished:
            while finished:
                yield {**finished.pop(0), "models": models}
            if not active:
                fill()
                continue
            groups: dict[int, tuple[Backend, list]] = {}
            for eid, (_, (role, messages, output_type), n) in active.items():
                call_seed = None if seed is None else zlib.crc32(f"{seed}/{eid}/{n}".encode())
                backend = llms[role]
                groups.setdefault(id(backend), (backend, []))[1].append(
                    (eid, Request(messages, output_type, role_sampling[role], call_seed))
                )
            for eid, result in _generate_groups(list(groups.values())):
                state = active[eid]
                try:
                    state[1] = state[0].throw(result) if isinstance(result, GenerationError) else state[0].send(result)
                    state[2] += 1
                except StopIteration as stop:
                    del active[eid]
                    yield {**stop.value, "models": models}
            fill()

    return drive()
