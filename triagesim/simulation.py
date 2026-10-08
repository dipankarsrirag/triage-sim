"""
Episode loop and driver.

An episode is a generator: it yields (role, messages, output_type) for each LLM call and gets the
validated output back. `simulate` keeps up to `concurrency` episodes in flight. With HTTP backends
(OpenRouter, `vllm serve`) each episode sends its next call as soon as its last one returns, so no
episode waits for another and the server batches whatever is in flight. With an in-process vLLM
engine the episodes move in rounds: all their pending calls go to the engine as one batch.
"""

import random
import re
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Iterator, Optional, Sequence

from triagesim import prompts
from triagesim.cases import Case
from triagesim.llm import Backend, GenerationError, Request, generate
from triagesim.personas import NursePersona, PatientPersona
from triagesim.schemas import (
    RECORD_FIELDS,
    RECORD_UNKNOWN,
    VITALS,
    Appearance,
    InSitu,
    LineEdit,
    PatientLineEdit,
    PatientOutput,
    PatientScript,
    Reading,
    NurseVerdict,
    Verdict,
    nurse_output_type,
)

DEFAULT_SAMPLING = {
    "nurse": {"max_tokens": 1024},
    "patient": {"max_tokens": 512},
    "judge": {"max_tokens": 512, "temperature": 0.0},
    "script": {"max_tokens": 2048},
}
MAX_FLAGS_PER_LOG = 5
TRIAGE_LEVEL = re.compile(r"\b(ESI|ATS|triage (level|category)|level [1-5]|category [1-5])\b", re.I)
# values that carry no information: they never replace what the record already holds
PLACEHOLDER = re.compile(
    rf"\s*({re.escape(RECORD_UNKNOWN)}|unchanged|no change|same|not (yet )?(established|mentioned|asked)|null|n/?a)?\s*\.?\s*$",
    re.I,
)

Call = tuple[str, list[dict[str, str]], type]
EpisodeGen = Generator[Call, Any, dict]


@dataclass(frozen=True)
class Features:
    """Optional parts of the simulation: the dialogue master describes the patient's appearance for
    the nurse (`appearance`); each nurse step shows the nurse its last assessment (`belief_state`) and
    the master's warm/cold feedback on its last level against the ground truth (`feedback`)."""

    appearance: bool = False
    belief_state: bool = False
    feedback: bool = False


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
    edit: Optional[tuple[Callable[[str, str], list[dict[str, str]]], type]] = None,
):
    """Yield a speaker call; if a judge is set, check the utterance and regenerate with its critique.

    Returns (output, verification). Verification is None when nothing was checked (no judge, or the
    nurse chose an action other than speaking). An utterance that still fails after
    `max_regenerations` is written by the judge itself if `edit` (messages builder, output type) is
    given (edited=True), else kept and flagged with passed=False; the rejected drafts are kept.
    """
    rejected: list[dict] = []
    current = messages
    while True:
        out = yield speaker, current, output_type
        if judge_messages is None or getattr(out, "action", "utterance") != "utterance":
            return out, None
        try:
            verdict_type = NurseVerdict if speaker == "nurse" else Verdict
            verdict: Verdict = yield "judge", judge_messages(out.utterance), verdict_type
        except GenerationError as e:  # an unavailable judge must not end the dialogue
            return out, {"passed": None, "error": str(e), "rejected": rejected}
        if verdict.passed:
            return out, {"passed": True, "verdict": verdict.model_dump(), "rejected": rejected}
        if len(rejected) == max_regenerations:
            if edit is not None:
                edit_messages, edit_type = edit
                try:
                    fixed = yield "judge", edit_messages(out.utterance, verdict.critique), edit_type
                    rejected.append({"utterance": out.utterance, "verdict": verdict.model_dump()})
                    out = out.model_copy(update=fixed.model_dump())
                    return out, {"passed": True, "edited": True, "verdict": verdict.model_dump(), "rejected": rejected}
                except GenerationError:  # keep the speaker's line, flagged
                    pass
            return out, {"passed": False, "verdict": verdict.model_dump(), "rejected": rejected}
        rejected.append({"utterance": out.utterance, "verdict": verdict.model_dump()})
        current = prompts.with_feedback(messages, out.model_dump_json(), verdict.critique)
        output_type = regenerate_as  # a regenerated nurse line must still be an utterance


def _nurse_phase(
    ep: _Episode,
    history: list,
    trace: list,
    red_flags: list,
    turn: int,
    questions_left: int,
    questions_before_end: int,
    record: dict,
    review: bool,
    max_regen: int,
    features: Features = Features(),
):
    """Nurse micro-steps until it speaks or ends triage; returns True if it ended.

    Within a phase the nurse may check one vital not yet known and log red flags once before
    speaking, so vital signs are spread through the conversation; only speaking uses up a question.
    With no questions left it may check every vital it still needs and log red flags, then must end;
    it may end only after `questions_before_end` more questions and, with a dialogue master
    (`review`), once every field of its triage record holds an answer. A vital the case lacks cannot be
    measured: checking it says so, and the nurse's next step must be a question that lets it infer
    that vital from the patient (even with no questions left). Its first step after a patient reply
    records what it understood from that reply on the reply's entry. Each step shows the nurse its
    triage record and how many questions it has left. What it says while taking a vital sign goes
    into the dialogue if it is longer than two words.
    """
    vital_checked = flag_logged = False
    deduce = None  # a vital that could not be measured: the nurse's next step must ask about it
    unread = history[-1] if history and history[-1]["actor"] == "patient" else None
    while True:
        reading = unread is not None
        known = {h["name"] for h in history if h.get("event") == "vital"}
        # with a dialogue master, the nurse may end before its questions run out only with a full record
        gaps = tuple(k for k in RECORD_FIELDS if record[k] == RECORD_UNKNOWN) if review and questions_left > 0 else ()
        if deduce:
            actions, vitals = ("utterance",), ()
        else:
            vitals = () if vital_checked and questions_left > 0 else tuple(v for v in VITALS if v not in known)
            actions = (
                (("utterance",) if questions_left > 0 else ())
                + (("check_vital",) if vitals else ())
                + (() if flag_logged else ("log_red_flag",))
                + (("end",) if questions_before_end <= 0 and not gaps else ())
            )
        belief = trace[-1] if features.belief_state and trace else None
        feedback = (prompts.warmth([t["triage"] for t in trace], ep.case.acuity)
                    if features.feedback and trace and ep.case.acuity is not None else None)
        messages = prompts.nurse_messages(
            ep.nurse_persona, ep.algorithm, ep.patient_view, history, actions, vitals, questions_left,
            questions_before_end, record, deduce, gaps, belief, feedback,
        )

        context = {"algorithm": ep.algorithm, "patient": ep.patient_view}

        def judge(utterance: str) -> list[dict[str, str]]:
            return prompts.judge_messages("nurse", utterance, ep.nurse_persona, history, **context)

        def editor(utterance: str, critique: str) -> list[dict[str, str]]:
            return prompts.edit_messages("nurse", utterance, critique, ep.nurse_persona, history, **context)

        out, verification = yield from _reviewed(
            "nurse",
            messages,
            nurse_output_type(actions, vitals, reading),
            judge if review else None,
            max_regen,
            nurse_output_type(("utterance",), (), reading),
            (editor, LineEdit),
        )
        trace.append({"turn": turn, **out.model_dump(), **({"feedback": feedback} if feedback else {})})
        if reading:
            unread["information"]["understood"] = out.understood
            unread = None

        if out.action == "check_vital":
            value = ep.case.vitals.get(out.vital)
            if out.utterance and len(out.utterance.split()) > 2:
                line = {"turn": turn, "actor": "nurse", "utterance": out.utterance, "action": "check_vital"}
                if value is None:  # the agents see the attempt; the output transcript leaves it out
                    line["in_transcript"] = False
                history.append(line)
            event = {"turn": turn, "actor": "system", "event": "vital", "name": out.vital, "value": value}
            if value is None:  # not in the case: cannot be measured, so the nurse asks about it next
                event["measurable"] = False
                deduce = out.vital
            history.append(event)
            vital_checked = True
        elif out.action == "log_red_flag":  # an empty list logs nothing; the step is used up
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
            if out.utterance and len(out.utterance.split()) > 2:  # the nurse's closing words to the patient
                line = {"turn": turn, "actor": "nurse", "utterance": out.utterance, "action": "end"}
                if TRIAGE_LEVEL.search(out.utterance):  # must not tell the patient its level: kept out of the output
                    line["in_transcript"] = False
                history.append(line)
            history.append({"turn": turn, "actor": "system", "event": "triage_end", "triage": out.triage})
            return True
        else:
            entry = {"turn": turn, "actor": "nurse", "utterance": out.utterance, "triage": out.triage}
            if deduce:
                entry["infers"] = deduce
            if verification is not None:
                entry["verification"] = verification
            history.append(entry)
            return False


def _run_episode(ep: _Episode, max_turns: int, min_questions: int, review: bool, max_regen: int,
                 features: Features = Features()) -> EpisodeGen:
    history: list[dict] = []
    trace: list[dict] = []
    red_flags: list[str] = []
    record = dict.fromkeys(RECORD_FIELDS, RECORD_UNKNOWN)  # the nurse's beliefs, kept by the master
    turn, error = 0, None
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
    if review and features.appearance:  # what the nurse sees at the desk; without it, only who and how they came
        messages = prompts.appearance_messages(ep.case, ep.patient_persona, script)
        try:
            for attempt in range(2):
                look = yield "script", messages, Appearance
                banned = sorted({m.group(0) for m in prompts.APPEARANCE_BANNED.finditer(" ".join(
                    v for v in look.model_dump().values() if v))})
                if not banned:
                    setup["appearance"] = look.model_dump()
                    ep.patient_view = prompts.patient_view(ep.case, ep.patient_persona, look)
                    break
                messages = prompts.with_feedback(messages, look.model_dump_json(),
                                                 f"Observations only: remove {', '.join(banned)}.")
            else:
                setup["appearance_error"] = f"kept judgement words or numbers: {', '.join(banned)}"
        except GenerationError as e:
            setup["appearance_error"] = str(e)

    try:
        while True:  # after the last question the nurse may still check vitals (and infer missing ones), then ends
            ended = yield from _nurse_phase(
                ep, history, trace, red_flags, turn, max(max_turns - turn, 0), min(min_questions, max_turns) - turn,
                record, review, max_regen, features,
            )
            if ended:
                break

            context = {"case": ep.case, "script": script}

            def judge(utterance: str) -> list[dict[str, str]]:
                return prompts.judge_messages("patient", utterance, ep.patient_persona, history, **context)

            def editor(utterance: str, critique: str) -> list[dict[str, str]]:
                return prompts.edit_messages("patient", utterance, critique, ep.patient_persona, history, **context)

            question = history[-1]["utterance"]
            reply, verification = yield from _reviewed(
                "patient",
                prompts.patient_messages(ep.patient_persona, ep.case, script, history),
                PatientOutput,
                judge if review else None,
                max_regen,
                PatientOutput,
                (editor, PatientLineEdit),
            )
            # what passed: the patient's own account, the master's reading of the exchange's words alone
            # (and what it added to the record), and (at the nurse's next step) what the nurse understood
            info = {"disclosed": reply.disclosed, "conveyed": None, "recorded": None, "understood": None}
            entry = {"turn": turn, "actor": "patient", "utterance": reply.utterance, "information": info}
            if verification is not None:
                entry["verification"] = verification
            history.append(entry)
            if review:
                try:  # a failed reading must not end the dialogue; the record just misses this exchange
                    reading = yield "judge", prompts.reading_messages(record, question, reply.utterance), Reading
                    info["conveyed"] = reading.conveyed
                    # the master rewrites the full record; a field changes only to real information
                    new = {k: v.strip() for k, v in reading.record.model_dump().items()}
                    info["recorded"] = {k: v for k, v in new.items() if not PLACEHOLDER.match(v) and v != record[k]}
                    record.update(info["recorded"])
                except GenerationError as e:
                    info["conveyed_error"] = str(e)
            turn += 1
    except GenerationError as e:
        error = str(e)

    return artifact(
        "error" if error else "ok",
        error=error,
        end_reason="error" if error else "max_turns" if turn >= max_turns else "nurse_end",
        final_triage=trace[-1]["triage"] if trace else None,
        num_turns=sum(h["actor"] == "patient" for h in history),
        transcript=prompts.spoken_transcript(history),
        history=history,
        trace=trace,
        red_flags=red_flags,
        record={**record, "vitals": {h["name"]: h["value"] for h in history if h.get("event") == "vital"}},
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
    judge_llm: Optional[Backend] = None,
    max_regenerations: int = 1,
    max_turns: int = 12,
    min_questions: int = 3,
    episodes_per_case: int = 1,
    seed: Optional[int] = 0,
    concurrency: int = 256,
    sampling: Optional[dict[str, dict]] = None,
    skip: Iterable[str] = (),
    features: Features = Features(),
) -> Iterator[dict]:
    """Simulate triage episodes, yielding one artifact dict per episode as it finishes.

    The nurse asks at most `max_turns` questions and may end triage only after `min_questions`.
    Episode ids are "<case_id>-<k>" for k < episodes_per_case. Personas (the patient taking the
    case's gender) are drawn per episode from `seed` and the episode id, so a run is reproducible and
    ids in `skip` (e.g. already done) can be left out without changing the rest.

    `algorithm` ("esi"/"ats") applies to every case; by default each case uses its `acuity_scale`,
    else ESI. With a `judge_llm` (the dialogue master), each episode starts with the master checking
    that the case would have a triage conversation in situ (status "skipped" with its reasoning if
    not) and writing the patient's script from the case and the patient persona; every spoken line is
    then checked and regenerated with feedback up to `max_regenerations` times, then flagged, and
    after every patient reply the master updates the nurse's triage record from the exchange (the
    nurse sees the record at every step; vital signs are added as they are taken). Cases without a
    ground-truth acuity are skipped. `sampling` overrides DEFAULT_SAMPLING per role ("nurse",
    "patient", "judge", "script"). Episodes whose LLM calls keep failing are yielded with status "error" and the
    dialogue up to the failure. `features` switches on the appearance (needs a dialogue master), the
    nurse's belief state and the warm/cold feedback (logged on each nurse step as "feedback").
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
        "judge": judge_llm,
        "script": judge_llm,
    }
    role_sampling = {role: {**DEFAULT_SAMPLING[role], **(sampling or {}).get(role, {})} for role in llms}
    models = {"nurse": nurse_llm.model, "patient": patient_llm.model}
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

        def fill() -> list[str]:
            """Start episodes up to `concurrency`; returns the ids of those now waiting on a call."""
            started = []
            while len(active) < concurrency and (ep := next(todo, None)) is not None:
                gen = _run_episode(ep, max_turns, min_questions, review, max_regenerations, features)
                try:
                    active[ep.episode_id] = [gen, next(gen), 0]
                    started.append(ep.episode_id)
                except StopIteration as stop:
                    finished.append(stop.value)
            return started

        def call(eid: str) -> tuple[Backend, Request]:
            _, (role, messages, output_type), n = active[eid]
            call_seed = None if seed is None else zlib.crc32(f"{seed}/{eid}/{n}".encode())
            return llms[role], Request(messages, output_type, role_sampling[role], call_seed)

        def advance(eid: str, result) -> Optional[dict]:
            """Send a call's result to its episode; returns the artifact if the episode ended."""
            state = active[eid]
            try:
                state[1] = state[0].throw(result) if isinstance(result, GenerationError) else state[0].send(result)
                state[2] += 1
            except StopIteration as stop:
                del active[eid]
                return {**stop.value, "models": models}
            return None

        def rounds() -> Iterator[dict]:
            fill()
            while active or finished:
                while finished:
                    yield {**finished.pop(0), "models": models}
                if not active:
                    fill()
                    continue
                groups: dict[int, tuple[Backend, list]] = {}
                for eid in active:
                    backend, request = call(eid)
                    groups.setdefault(id(backend), (backend, []))[1].append((eid, request))
                for eid, result in _generate_groups(list(groups.values())):
                    if (done := advance(eid, result)) is not None:
                        yield done
                fill()

        def independent() -> Iterator[dict]:
            pool = ThreadPoolExecutor(concurrency)
            pending: dict[Future, str] = {}

            def submit(eids: list[str]) -> None:
                for eid in eids:
                    backend, request = call(eid)
                    pending[pool.submit(lambda b=backend, r=request: generate(b, [r])[0])] = eid

            try:
                submit(fill())
                while pending or finished:
                    while finished:
                        yield {**finished.pop(0), "models": models}
                    if not pending:
                        submit(fill())
                        continue
                    for future in wait(pending, return_when=FIRST_COMPLETED).done:
                        eid = pending.pop(future)
                        if (done := advance(eid, future.result())) is None:
                            submit([eid])
                        else:
                            yield done
                    submit(fill())
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        batched = any(b.batched for b in llms.values() if b is not None)
        return rounds() if batched else independent()

    return drive()
