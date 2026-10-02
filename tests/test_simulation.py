import pytest
from conftest import FakeBackend, allowed, nurse_says

from triagesim.cases import Case
from triagesim.schemas import PatientOutput, PatientScript
from triagesim.simulation import simulate


def run(cases, backend, nurse_persona, patient_persona, **kwargs):
    kwargs.setdefault("patient_llm", backend)
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
        "vital", "nurse", "patient",  # turn 0: check pulse, log a flag, ask
        "nurse", "patient",           # turn 1: re-log (deduplicated), ask
        "triage_end",                 # turn 2: re-log, end
    ]
    assert ep["history"][0] == {"turn": 0, "actor": "system", "event": "vital", "name": "heartrate", "value": 112}
    assert ep["red_flags"] == ["Tachycardia"]  # deduplicated across entries and phases
    assert [b["turn"] for b in ep["beliefs"]] == [0, 1]
    assert [t["action"] for t in ep["trace"]] == [
        "check_vital", "log_red_flag", "utterance", "log_red_flag", "utterance", "log_red_flag", "end"
    ]
    assert ep["models"] == {"nurse": "fake", "patient": "fake", "belief": "fake"}


def test_actions_are_masked_within_a_phase(backend, case, nurse_persona, patient_persona):
    run([case], backend, nurse_persona, patient_persona)
    nurse_requests = [r for batch in backend.batches for r in batch if "vital" in r.output_type.model_fields]
    first, second, third = nurse_requests[:3]
    assert "check_vital" in allowed(first, "action")
    assert "check_vital" not in allowed(second, "action")  # one vital per phase
    assert "log_red_flag" not in allowed(third, "action")  # one log per phase
    # the next phase may check a vital again, but not one already known
    fourth = nurse_requests[3]
    assert "check_vital" in allowed(fourth, "action") and "heartrate" not in allowed(fourth, "vital")


def test_max_turns_without_nurse_ending(case, nurse_persona, patient_persona):
    chatty = FakeBackend(nurse=lambda r: nurse_says(r, action="utterance", utterance="And then?", triage=4))
    [ep] = run([case], chatty, nurse_persona, patient_persona, max_turns=3, extract_beliefs=False)
    assert (ep["end_reason"], ep["num_turns"], ep["final_triage"]) == ("max_turns", 3, 4)
    assert ep["beliefs"] == [] and "belief" not in ep["models"]


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
                       patient_personas=[patient_persona], extract_beliefs=False, **kwargs)
        return {e["episode_id"]: e["nurse_persona"]["gender"] for e in eps}

    full = personas(seed=1)
    assert full == personas(seed=1)
    assert personas(seed=1, skip={"0-0", "3-0"}) == {k: v for k, v in full.items() if k not in {"0-0", "3-0"}}


def test_roles_can_use_different_backends(case, nurse_persona, patient_persona):
    nurse, patient = FakeBackend(model="nurse-llm"), FakeBackend(model="patient-llm")
    [ep] = run([case], nurse, nurse_persona, patient_persona, patient_llm=patient)
    assert ep["status"] == "ok"
    assert ep["models"] == {"nurse": "nurse-llm", "patient": "patient-llm", "belief": "nurse-llm"}
    assert patient.batches and all(r.output_type is PatientOutput for b in patient.batches for r in b)


def test_bad_arguments_fail_immediately(backend, case, nurse_persona, patient_persona):
    with pytest.raises(ValueError):
        simulate([case], nurse_llm=backend, patient_llm=backend, nurse_personas=[nurse_persona],
                 patient_personas=[patient_persona], algorithm="mts")
    with pytest.raises(ValueError):
        simulate([case], nurse_llm=backend, patient_llm=backend, nurse_personas=[], patient_personas=[patient_persona])


def test_failed_belief_extraction_keeps_the_dialogue_going(case, nurse_persona, patient_persona):
    flaky = FakeBackend(belief=lambda r: "not json")
    [ep] = run([case], flaky, nurse_persona, patient_persona)
    assert ep["status"] == "ok" and ep["end_reason"] == "nurse_end"
    assert [set(b) for b in ep["beliefs"]] == [{"turn", "error"}] * 2


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
    eps = {e["episode_id"]: e for e in run(cases, backend, nurse_persona, patient_persona, extract_beliefs=False)}
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
        return {"critique": "ok" if ok else "invented a diagnosis", "faithful": ok, "informative": True, "in_persona": True}

    def patient(request):
        return {"utterance": "fixed reply" if len(request.messages) > 2 else "I have a stroke.", "disclosed": "x"}

    backend = FakeBackend(judge=judge, patient=patient)
    [ep] = run([case], backend, nurse_persona, patient_persona, judge_llm=backend, extract_beliefs=False)
    replies = [h for h in ep["history"] if h["actor"] == "patient"]
    assert replies[0]["utterance"] == "fixed reply"
    check = replies[0]["verification"]
    assert check["passed"] is True and check["rejected"][0]["utterance"] == "I have a stroke."
    retry = next(r for b in backend.batches for r in b if r.output_type is PatientOutput and len(r.messages) > 2)
    assert retry.messages[-2]["role"] == "assistant" and "invented a diagnosis" in retry.messages[-1]["content"]
    assert ep["models"]["judge"] == "fake"


def test_lines_that_keep_failing_are_flagged_and_kept(case, nurse_persona, patient_persona):
    never = FakeBackend(judge=lambda r: {"critique": "off persona", "faithful": True, "informative": True, "in_persona": False})
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
    for reply in replies:
        assert reply["information"] == {
            "disclosed": "I fainted this morning.",             # the patient's own account
            "conveyed": "The patient fainted this morning.",    # the master, from the words alone
            "understood": "The patient fainted.",               # the nurse, reading it at its next step
        }
    # the nurse writes `understood` only in its first step after a reply
    readings = [t["understood"] for t in ep["trace"]]
    assert readings[0] is None  # nothing to read before the patient speaks
    assert sum(r is not None for r in readings) == len(replies)
    # the master reads the utterance alone: no case, history or script in its prompt
    reads = [r for b in backend.batches for r in b if r.output_type.__name__ == "Reading"]
    assert reads and all(r.messages[1]["content"] == 'Patient: """I fainted this morning."""' for r in reads)


def test_without_a_dialogue_master_only_the_agents_log(backend, case, nurse_persona, patient_persona):
    [ep] = run([case], backend, nurse_persona, patient_persona)
    info = next(h for h in ep["history"] if h["actor"] == "patient")["information"]
    assert info["disclosed"] and info["understood"] and info["conveyed"] is None
