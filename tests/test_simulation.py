import threading
import zlib

import pytest
from conftest import FakeBackend, allowed, nurse_policy, nurse_says, verdict

from triagesim.cases import Case
from triagesim.schemas import RECORD_FIELDS, RECORD_UNKNOWN, PatientOutput, PatientScript
from triagesim.simulation import simulate


def run(cases, backend, nurse_persona, patient_persona, **kwargs):
    kwargs.setdefault("patient_llm", backend)
    kwargs.setdefault("min_questions", 0)  # the fake nurse ends after two replies; see test_minimum_questions
    return list(
        simulate(
            cases,
            nurse_llm=backend,
            nurse_personas=[nurse_persona],
            patient_personas=[patient_persona],
            **kwargs,
        )
    )


def test_full_episode(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona)

    assert ep["status"] == "ok" and ep["error"] is None
    assert ep["episode_id"] == "c1-0"
    assert (ep["end_reason"], ep["final_triage"], ep["num_turns"]) == ("nurse_end", 2, 2)
    assert [h.get("event") or h["actor"] for h in ep["history"]] == [
        "nurse", "vital", "nurse", "patient",  # turn 0: say and check pulse, log a flag, ask
        "nurse", "patient",                    # turn 1: re-log (deduplicated), ask
        "triage_end",                          # turn 2: re-log, end
    ]
    assert ep["history"][0] == {"turn": 0, "actor": "nurse", "utterance": "Let me check your pulse.", "action": "check_vital"}
    assert [t["speaker"] for t in ep["transcript"]] == ["nurse", "nurse", "patient", "nurse", "patient"]  # spoken lines only
    assert ep["history"][1] == {"turn": 0, "actor": "system", "event": "vital", "name": "heartrate", "value": 112}
    assert ep["red_flags"] == ["Tachycardia"]  # deduplicated across entries and phases
    # without a dialogue master the record holds only the vital signs taken
    assert ep["record"] == {**dict.fromkeys(RECORD_FIELDS, RECORD_UNKNOWN), "vitals": {"heartrate": 112}}
    assert [t["action"] for t in ep["trace"]] == [
        "check_vital", "log_red_flag", "utterance", "log_red_flag", "utterance", "log_red_flag", "end"
    ]
    assert ep["models"] == {"nurse": "fake", "patient": "fake"}


def test_nurse_is_told_how_many_questions_are_left(backend, case, nurse_persona, patient_persona):
    run([case], backend, nurse_persona, patient_persona, max_turns=3)
    nurse_requests = [r for batch in backend.batches for r in batch if "vital" in r.output_type.model_fields]
    left = [int(r.messages[1]["content"].split("You can ask ")[1].split()[0]) for r in nurse_requests]
    assert left == [3, 3, 3, 2, 2, 1, 1]  # per step: phases 1-2 ask, phase 3 ends; vitals and flags are free


def test_minimum_questions_before_the_nurse_may_end(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona, min_questions=3)
    assert (ep["end_reason"], ep["num_turns"]) == ("nurse_end", 3)  # wanted to end after 2, had to ask a third
    nurse_requests = [r for batch in backend.batches for r in batch if "vital" in r.output_type.model_fields]
    phases = [r for r in nurse_requests if "end" not in allowed(r, "action")]
    assert phases and all("You must ask at least" in r.messages[1]["content"] for r in phases)
    assert "end" in allowed(nurse_requests[-1], "action")


def test_actions_are_masked_within_a_phase(backend, case, nurse_persona, patient_persona):
    run([case], backend, nurse_persona, patient_persona)
    nurse_requests = [r for batch in backend.batches for r in batch if "vital" in r.output_type.model_fields]
    first, second, third = nurse_requests[:3]
    assert "check_vital" in allowed(first, "action")
    assert "check_vital" not in allowed(second, "action")  # one vital per question: spread through the dialogue
    assert "log_red_flag" not in allowed(third, "action")  # one log per phase
    # the next phase may check a vital again, but not one already known
    fourth = nurse_requests[3]
    assert "check_vital" in allowed(fourth, "action") and "heartrate" not in allowed(fourth, "vital")


def test_after_the_last_question_every_missing_vital_may_be_taken(case, nurse_persona, patient_persona):
    def chatty(r):  # asks while it can; then takes every vital it may, then ends
        actions = allowed(r, "action")
        if "utterance" in actions:
            return nurse_says(r, action="utterance", utterance="And then?")
        if "check_vital" in actions:
            return nurse_says(r, action="check_vital", vital=allowed(r, "vital")[0], utterance="Let me check this.")
        return nurse_says(r, action="end")

    [ep] = run([case], FakeBackend(nurse=chatty), nurse_persona, patient_persona, max_turns=2)
    taken = [h["name"] for h in ep["history"] if h.get("event") == "vital"]
    assert taken[:2] == ["temperature", "heartrate"]  # one before each of the two questions
    assert sorted(taken) == sorted(["temperature", "heartrate", "resprate", "o2sat", "sbp"])  # the rest at the end


def test_max_turns_without_nurse_ending(case, nurse_persona, patient_persona):
    def chatty(r):  # always asks; once no questions are left it can only end
        if "utterance" in allowed(r, "action"):
            return nurse_says(r, action="utterance", utterance="And then?", triage=4)
        return nurse_says(r, action="end", triage=3)

    backend = FakeBackend(nurse=chatty)
    [ep] = run([case], backend, nurse_persona, patient_persona, max_turns=3)
    assert (ep["end_reason"], ep["num_turns"], ep["final_triage"]) == ("max_turns", 3, 3)
    # after the last reply the nurse gets a final step: it reads the reply and ends with its level
    assert ep["trace"][-1]["action"] == "end" and ep["trace"][-1]["turn"] == 3
    assert ep["history"][-1] == {"turn": 3, "actor": "system", "event": "triage_end", "triage": 3}
    assert ep["history"][-2]["information"]["understood"] == "The patient fainted."
    last = [r for b in backend.batches for r in b if "vital" in r.output_type.model_fields][-1]
    assert "no questions left" in last.messages[1]["content"]


def test_failed_generation_yields_partial_error_episode(case, nurse_persona, patient_persona):
    mute = FakeBackend(patient=lambda r: '{"utterance": "", "disclosed": "nothing"}')
    [ep] = run([case], mute, nurse_persona, patient_persona)
    assert ep["status"] == "error" and ep["end_reason"] == "error"
    assert "utterance" in ep["error"]
    assert ep["history"][-1]["actor"] == "nurse"  # dialogue up to the failure is kept


def test_episodes_are_batched_up_to_concurrency(backend, nurse_persona, patient_persona):
    cases = [Case(case_id=str(i), chief_complaint="Cough", acuity=3) for i in range(5)]
    eps = run(cases, backend, nurse_persona, patient_persona, concurrency=2, episodes_per_case=2)
    assert sorted(e["episode_id"] for e in eps) == sorted(f"{i}-{k}" for i in range(5) for k in range(2))
    assert max(len(b) for b in backend.batches) == 2

    wide = FakeBackend()
    run(cases, wide, nurse_persona, patient_persona)
    assert len(wide.batches[0]) == 5


def test_persona_sampling_is_seeded_and_skip_leaves_others_unchanged(backend, nurse_persona, patient_persona):
    cases = [Case(case_id=str(i), chief_complaint="Cough", acuity=3) for i in range(6)]
    nurses = [nurse_persona.model_copy(update={"gender": g}) for g in ("a", "b", "c")]

    def personas(**kwargs):
        eps = simulate(cases, nurse_llm=backend, patient_llm=backend, nurse_personas=nurses,
                       patient_personas=[patient_persona], **kwargs)
        return {e["episode_id"]: e["nurse_persona"]["gender"] for e in eps}

    full = personas(seed=1)
    assert full == personas(seed=1)
    assert personas(seed=1, skip={"0-0", "3-0"}) == {k: v for k, v in full.items() if k not in {"0-0", "3-0"}}


def test_roles_can_use_different_backends(case, nurse_persona, patient_persona):
    nurse, patient = FakeBackend(model="nurse-llm"), FakeBackend(model="patient-llm")
    [ep] = run([case], nurse, nurse_persona, patient_persona, patient_llm=patient)
    assert ep["status"] == "ok"
    assert ep["models"] == {"nurse": "nurse-llm", "patient": "patient-llm"}
    assert patient.batches and all(r.output_type is PatientOutput for b in patient.batches for r in b)


def test_bad_arguments_fail_immediately(backend, case, nurse_persona, patient_persona):
    with pytest.raises(ValueError):
        simulate([case], nurse_llm=backend, patient_llm=backend, nurse_personas=[nurse_persona],
                 patient_personas=[patient_persona], algorithm="mts")
    with pytest.raises(ValueError):
        simulate([case], nurse_llm=backend, patient_llm=backend, nurse_personas=[], patient_personas=[patient_persona])


def test_failed_reading_keeps_the_dialogue_going(case, nurse_persona, patient_persona):
    flaky = FakeBackend(reader=lambda r: "not json")
    [ep] = run([case], flaky, nurse_persona, patient_persona, judge_llm=flaky, max_turns=4)
    assert ep["status"] == "ok" and ep["end_reason"] == "max_turns"  # no record: the nurse cannot end early
    replies = [h for h in ep["history"] if h["actor"] == "patient"]
    assert all(r["information"]["conveyed_error"] and r["information"]["recorded"] is None for r in replies)
    assert ep["record"]["chief_complaint"] == RECORD_UNKNOWN


def test_placeholder_record_values_do_not_overwrite_the_record(case, nurse_persona, patient_persona):
    def reader(r):  # the master rewrites the full record; later it writes placeholders over known fields
        first = f"chief_complaint: {RECORD_UNKNOWN}" in r.messages[1]["content"]
        record = dict.fromkeys(RECORD_FIELDS, RECORD_UNKNOWN)
        if first:
            record |= {"chief_complaint": "fainted", "allergies": "none known"}
        else:
            record |= {"chief_complaint": "unchanged", "allergies": RECORD_UNKNOWN.capitalize() + ".", "medications": "none"}
        return {"conveyed": "x", "record": record}

    backend = FakeBackend(reader=reader)
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend)
    assert ep["record"]["chief_complaint"] == "fainted" and ep["record"]["allergies"] == "none known"
    assert ep["record"]["medications"] == "none"  # "none" is information, not a placeholder
    assert ep["record"]["pain"] == RECORD_UNKNOWN


def test_an_unmeasurable_vital_forces_a_question_to_infer_it(case, nurse_persona, patient_persona):
    no_pulse = case.model_copy(update={"vitals": {**case.vitals, "heartrate": None}})
    backend = FakeBackend()
    [ep] = run([no_pulse], backend, nurse_persona, patient_persona)
    event = next(h for h in ep["history"] if h.get("event") == "vital")
    assert event["name"] == "heartrate" and event["value"] is None and event["measurable"] is False
    nurse_steps = [r for b in backend.batches for r in b if r.output_type.__name__.startswith("NurseOutput")]
    forced = nurse_steps[1]  # the step right after the failed measurement
    assert allowed(forced, "action") == ["utterance"]
    assert "The heartrate could not be measured. Your next action must be a question" in forced.messages[1]["content"]
    question = next(h for h in ep["history"] if h["actor"] == "nurse" and h.get("action") != "check_vital")
    assert question["infers"] == "heartrate"
    assert "[Vital] heartrate: could not be measured" in nurse_steps[-1].messages[1]["content"]
    # the announcement stays in the history (and the agents' prompts) but not in the output transcript
    announced = next(h for h in ep["history"] if h.get("action") == "check_vital")
    assert announced["in_transcript"] is False
    assert "Nurse: Let me check your pulse." in nurse_steps[-1].messages[1]["content"]
    assert {"speaker": "nurse", "text": "Let me check your pulse."} not in ep["transcript"]
    assert ep["transcript"][0] == {"speaker": "nurse", "text": question["utterance"]}
    assert "- heartrate: could not be measured; infer it from the patient's answers" in nurse_steps[-1].messages[1]["content"]


def test_an_unmeasurable_vital_after_the_last_question_still_gets_its_question(case, nurse_persona, patient_persona):
    def late(r):  # asks while it can, then tries the pulse, then ends
        actions = allowed(r, "action")
        if "utterance" in actions and "check_vital" not in actions and "end" not in actions:
            return nurse_says(r, action="utterance", utterance="Any palpitations or dizziness?")  # forced
        if "utterance" in actions:
            return nurse_says(r, action="utterance", utterance="And then?")
        if "heartrate" in allowed(r, "vital"):
            return nurse_says(r, action="check_vital", vital="heartrate", utterance="Let me check your pulse.")
        return nurse_says(r, action="end")

    no_pulse = case.model_copy(update={"vitals": {**case.vitals, "heartrate": None}})
    [ep] = run([no_pulse], FakeBackend(nurse=late), nurse_persona, patient_persona, max_turns=1)
    asked = [h["utterance"] for h in ep["history"] if h["actor"] == "nurse" and h.get("action") != "check_vital"]
    assert asked == ["And then?", "Any palpitations or dizziness?"]  # one extra, forced question
    assert ep["num_turns"] == 2 and ep["end_reason"] == "max_turns"


def test_with_a_master_the_nurse_ends_only_with_a_full_record(case, nurse_persona, patient_persona):
    def partial(r):  # records only the complaint
        return {"conveyed": "x", "record": dict.fromkeys(RECORD_FIELDS, RECORD_UNKNOWN) | {"chief_complaint": "fainted"}}

    def full(r):  # every field answered, some with "declined to say"
        return {"conveyed": "x", "record": dict.fromkeys(RECORD_FIELDS, "declined to say") | {"chief_complaint": "fainted"}}

    gated = FakeBackend(reader=partial)
    [ep] = run([case], gated, nurse_persona, patient_persona, judge_llm=gated, max_turns=5)
    assert (ep["end_reason"], ep["num_turns"]) == ("max_turns", 5)  # wanted to end after 2, could not
    nurse_steps = [r for b in gated.batches for r in b if r.output_type.__name__.startswith("NurseOutput")]
    told = [r for r in nurse_steps if "still missing: onset and course, pain, associated symptoms" in r.messages[1]["content"]]
    assert told and all("end" not in allowed(r, "action") for r in told)

    done = FakeBackend(reader=full)
    [ep] = run([case], done, nurse_persona, patient_persona, judge_llm=done, max_turns=5)
    assert (ep["end_reason"], ep["num_turns"]) == ("nurse_end", 2)


def test_the_nurse_closes_with_a_spoken_line_that_never_gives_the_level(case, nurse_persona, patient_persona):
    def closing(words):
        def nurse(r):
            out = nurse_policy(r)
            return {**out, "utterance": words} if out["action"] == "end" else out
        return nurse

    [ep] = run([case], FakeBackend(nurse=closing("Thank you, please take a seat and we will call you soon.")),
               nurse_persona, patient_persona)
    assert ep["history"][-2] == {"turn": 2, "actor": "nurse", "action": "end",
                                 "utterance": "Thank you, please take a seat and we will call you soon."}
    assert ep["transcript"][-1] == {"speaker": "nurse", "text": "Thank you, please take a seat and we will call you soon."}
    [ep] = run([case], FakeBackend(nurse=closing("You are ESI 2, please wait.")), nurse_persona, patient_persona)
    assert ep["history"][-2]["in_transcript"] is False and ep["transcript"][-1]["speaker"] == "patient"


def test_short_vital_check_lines_stay_out_of_the_dialogue(case, nurse_persona, patient_persona):
    def terse(r):
        return {**nurse_policy(r), "utterance": "Pulse check."} if "heartrate" in allowed(r, "vital") else nurse_policy(r)

    [ep] = run([case], FakeBackend(nurse=terse), nurse_persona, patient_persona)
    assert ep["history"][0]["event"] == "vital"  # two words: not spoken in the transcript


def test_unknown_sampling_role_is_rejected(backend, case, nurse_persona, patient_persona):
    with pytest.raises(ValueError, match="nurses"):
        simulate([case], nurse_llm=backend, patient_llm=backend, nurse_personas=[nurse_persona],
                 patient_personas=[patient_persona], sampling={"nurses": {"temperature": 0}})


def test_sampling_overrides_reach_requests(backend, case, nurse_persona, patient_persona):
    run([case], backend, nurse_persona, patient_persona, sampling={"patient": {"temperature": 0.3}})
    patient_sampling = [r.sampling for b in backend.batches for r in b if r.output_type is PatientOutput]
    assert patient_sampling[0] == {"max_tokens": 512, "temperature": 0.3}



# ─────────────────────────────────────────
# Final decision, personas, scripts, dialogue master
# ─────────────────────────────────────────


def test_case_scale_picks_the_algorithm(backend, case, nurse_persona, patient_persona):
    [ep] = run([case.model_copy(update={"acuity_scale": "ats"})], backend, nurse_persona, patient_persona)
    assert ep["algorithm"] == "ats"
    assert "Australasian Triage Scale" in backend.batches[0][0].messages[0]["content"]
    [forced] = run([case.model_copy(update={"acuity_scale": "ats"})], backend, nurse_persona, patient_persona,
                   algorithm="esi")
    assert forced["algorithm"] == "esi"


def test_patient_gender_comes_from_the_case(backend, case, nurse_persona, patient_persona):
    cases = [case.model_copy(update={"case_id": "f", "gender": "female"}), case.model_copy(update={"case_id": "u"})]
    eps = {e["episode_id"]: e for e in run(cases, backend, nurse_persona, patient_persona)}
    assert eps["f-0"]["patient_persona"]["gender"] == "female"  # the persona drawn was male
    assert eps["u-0"]["patient_persona"]["gender"] == "male"  # no gender in the case: the persona's own


def test_master_writes_the_script_from_case_and_persona(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend)
    assert ep["status"] == "ok" and ep["in_situ"]["conversation"] is True
    assert ep["patient_script"]["story"] == "I fainted at the bus stop."
    order = [r.output_type.__name__ for b in backend.batches for r in b][:2]
    assert order == ["InSitu", "PatientScript"]  # before the dialogue starts
    script_call = next(r for b in backend.batches for r in b if r.output_type is PatientScript)
    assert "Syncope" in script_call.messages[1]["content"] and "PATIENT PERSONA" in script_call.messages[1]["content"]
    patient_prompt = next(r for b in backend.batches for r in b if r.output_type is PatientOutput).messages[0]["content"]
    assert "I fainted at the bus stop." in patient_prompt and "chest pain" in patient_prompt


def test_master_skips_cases_without_an_in_situ_conversation(case, nurse_persona, patient_persona):
    resus = FakeBackend(in_situ=lambda r: {"reasoning": "arrest on arrival: straight to resus", "conversation": False})
    [ep] = run([case], resus, nurse_persona, patient_persona, judge_llm=resus)
    assert (ep["status"], ep["skip_reason"]) == ("skipped", "arrest on arrival: straight to resus")
    assert [r.output_type.__name__ for b in resus.batches for r in b] == ["InSitu"]  # nothing else simulated


def test_cases_without_acuity_are_skipped_before_any_call(backend, case, nurse_persona, patient_persona):
    cases = [case.model_copy(update={"case_id": "x", "acuity": None}), case]
    eps = run(cases, backend, nurse_persona, patient_persona)
    assert {e["episode_id"]: e["status"] for e in eps} == {"x-0": "skipped", "c1-0": "ok"}
    assert next(e for e in eps if e["status"] == "skipped")["skip_reason"] == "no ground-truth acuity"


def test_without_a_master_there_is_no_check_or_script(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona)
    assert ep["patient_script"] is None and "in_situ" not in ep


def test_rejected_lines_are_regenerated_with_feedback(case, nurse_persona, patient_persona):
    def judge(request):  # rejects any line without "fixed"; the regenerated patient line has it
        ok = "fixed" in request.messages[1]["content"].rsplit("UTTERANCE", 1)[1]
        return verdict(request, "ok" if ok else "invented a diagnosis", faithful=ok)

    def patient(request):
        return {"utterance": "fixed reply" if len(request.messages) > 2 else "I have a stroke.", "disclosed": "x"}

    backend = FakeBackend(judge=judge, patient=patient)
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend)
    replies = [h for h in ep["history"] if h["actor"] == "patient"]
    assert replies[0]["utterance"] == "fixed reply"
    check = replies[0]["verification"]
    assert check["passed"] is True and check["rejected"][0]["utterance"] == "I have a stroke."
    retry = next(r for b in backend.batches for r in b if r.output_type is PatientOutput and len(r.messages) > 2)
    assert retry.messages[-2]["role"] == "assistant" and "invented a diagnosis" in retry.messages[-1]["content"]
    assert ep["models"]["judge"] == "fake"


def never_passes(r):
    return verdict(r, "nobody would say that", plausible=False)


def test_lines_that_keep_failing_are_written_by_the_master(case, nurse_persona, patient_persona):
    never = FakeBackend(judge=never_passes)
    [ep] = run([case], never, nurse_persona, patient_persona, judge_llm=never, max_regenerations=1)
    assert ep["status"] == "ok"
    lines = [h for h in ep["history"] if "verification" in h]
    assert lines and all(h["utterance"] == "Edited by the master." for h in lines)
    checks = [h["verification"] for h in lines]
    assert all(c["passed"] is True and c["edited"] is True and len(c["rejected"]) == 2 for c in checks)
    assert all(c["verdict"]["critique"] == "nobody would say that" for c in checks)
    patient = next(h for h in lines if h["actor"] == "patient")
    assert patient["information"]["disclosed"] == "x"  # the master's version says what it reveals
    edits = [r for b in never.batches for r in b if r.output_type.__name__ in ("LineEdit", "PatientLineEdit")]
    assert edits and "nobody would say that" in edits[0].messages[0]["content"]


def test_lines_that_keep_failing_are_flagged_if_the_master_cannot_edit(case, nurse_persona, patient_persona):
    never = FakeBackend(judge=never_passes, editor=lambda r: "not json")
    [ep] = run([case], never, nurse_persona, patient_persona, judge_llm=never, max_regenerations=1)
    assert ep["status"] == "ok"
    checks = [h["verification"] for h in ep["history"] if "verification" in h]
    assert checks and all(c["passed"] is False and len(c["rejected"]) == 1 for c in checks)
    # nurse micro-actions (vital checks, red flags, end) are not reviewed
    assert all("verification" not in h for h in ep["history"] if h["actor"] == "system")


def test_information_flow_is_logged_on_each_patient_line(case, nurse_persona, patient_persona):
    backend = FakeBackend()
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend)
    replies = [h for h in ep["history"] if h["actor"] == "patient"]
    for i, reply in enumerate(replies):
        assert reply["information"] == {
            "disclosed": "I fainted this morning.",             # the patient's own account
            "conveyed": "The patient fainted this morning.",    # the master, from the exchange's words alone
            # what changed in the record (the second reply repeats what is already recorded)
            "recorded": {"chief_complaint": "fainted this morning"} if i == 0 else {},
            "understood": "The patient fainted.",               # the nurse, reading it at its next step
        }
    # the nurse writes `understood` only in its first step after a reply
    readings = [t["understood"] for t in ep["trace"]]
    assert readings[0] is None  # nothing to read before the patient speaks
    assert sum(r is not None for r in readings) == len(replies)
    # the master reads the exchange alone (record so far + question + reply): no case or script
    reads = [r for b in backend.batches for r in b if r.output_type.__name__ == "Reading"]
    assert reads and all('Nurse: """Hello. What brings you in today?"""\nPatient: """I fainted this morning."""'
                         in r.messages[1]["content"] and "Syncope" not in r.messages[1]["content"] for r in reads)
    assert f"chief_complaint: {RECORD_UNKNOWN}" in reads[0].messages[1]["content"]
    assert "chief_complaint: fainted this morning" in reads[1].messages[1]["content"]
    # the nurse sees its record, kept by the master, at every step after the first reply
    assert ep["record"]["chief_complaint"] == "fainted this morning" and ep["record"]["vitals"] == {"heartrate": 112}
    nurse_steps = [r for b in backend.batches for r in b if r.output_type.__name__.startswith("NurseOutput")]
    assert f"chief complaint: {RECORD_UNKNOWN}" in nurse_steps[0].messages[1]["content"]
    assert "chief complaint: fainted this morning" in nurse_steps[-1].messages[1]["content"]
    assert "heartrate 112 bpm" in nurse_steps[-1].messages[1]["content"]


def test_without_a_dialogue_master_only_the_agents_log(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona)
    info = next(h for h in ep["history"] if h["actor"] == "patient")["information"]
    assert info["disclosed"] and info["understood"] and info["conveyed"] is None


class HTTPBackend(FakeBackend):
    batched = False  # driven one call at a time per episode, like OpenRouter or `vllm serve`


def test_http_backends_never_make_an_episode_wait_for_another(nurse_persona, patient_persona):
    cases = [Case(case_id=c, chief_complaint="Cough", acuity=3) for c in ("slow", "fast")]
    stuck = zlib.crc32(b"0/slow-0/0")  # the slow episode's first call
    release = threading.Event()

    class Backend(HTTPBackend):
        def complete(self, requests):
            if any(r.seed == stuck for r in requests):
                assert release.wait(10), "the fast episode waited for the slow one"
            return super().complete(requests)

    episodes = simulate(cases, nurse_llm=Backend(), patient_llm=Backend(), nurse_personas=[nurse_persona],
                        patient_personas=[patient_persona], min_questions=0)
    assert next(episodes)["episode_id"] == "fast-0"  # finished while "slow" was still on its first call
    release.set()
    assert next(episodes)["episode_id"] == "slow-0"


def test_http_and_round_driving_give_the_same_episodes(nurse_persona, patient_persona):
    cases = [Case(case_id=str(i), chief_complaint="Cough", acuity=3) for i in range(4)]

    def episodes(backend):
        eps = run(cases, backend, nurse_persona, patient_persona, judge_llm=backend, episodes_per_case=2,
                  concurrency=3)
        return {e["episode_id"]: e for e in eps}

    rounds, http = episodes(FakeBackend()), episodes(HTTPBackend())
    assert len(http) == 8 and http == rounds


def test_a_nurse_line_asking_two_things_is_regenerated(case, nurse_persona, patient_persona):
    double = "When did it start, and are you bleeding?"

    def nurse(request):  # asks two things at first; the regenerated line (with feedback) asks one
        out = nurse_policy(request)
        if out["action"] == "utterance" and len(request.messages) == 2:
            out["utterance"] = double
        return out

    def judge(request):
        line = request.messages[1]["content"].rsplit("UTTERANCE", 1)[1]
        return verdict(request, "asks two things" if double in line else "fine", one_question=double not in line)

    backend = FakeBackend(nurse=nurse, judge=judge)
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend)
    first = next(h for h in ep["history"] if h["actor"] == "nurse" and "verification" in h)
    assert first["utterance"] != double and first["verification"]["passed"] is True
    assert first["verification"]["rejected"][0]["verdict"]["one_question"] is False
    checks = [r.output_type.__name__ for b in backend.batches for r in b if r.output_type.__name__.endswith("Verdict")]
    assert "NurseVerdict" in checks and "Verdict" in checks  # only nurse lines get the one-question check
