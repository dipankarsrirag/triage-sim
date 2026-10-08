from triagesim import prompts


def test_no_one_uses_names_and_the_judge_checks_it(nurse_persona, patient_persona, case):
    nurse = prompts.nurse_messages(nurse_persona, "esi", "- gender: male", [], ("utterance", "end"), ())
    assert "Never use names" in nurse[0]["content"] and "name" not in nurse[1]["content"]
    assert "Never say a personal name" in prompts.patient_messages(patient_persona, case, None, [])[0]["content"]
    for speaker, persona in (("nurse", nurse_persona), ("patient", patient_persona)):
        judge = prompts.judge_messages(speaker, "Hi.", persona, [], case=case)
        assert "- no_names:" in judge[0]["content"]


def test_prompt_prefix_is_static_and_state_is_in_user_message(nurse_persona):
    early = prompts.nurse_messages(nurse_persona, "ats", "- gender: male", [], ("utterance", "check_vital", "end"), ("sbp",))
    later = prompts.nurse_messages(
        nurse_persona, "ats", "- gender: male",
        [{"turn": 0, "actor": "system", "event": "vital", "name": "sbp", "value": 98.0}],
        ("utterance", "end"), (),
    )
    assert early[0] == later[0]  # system prompt shared by every step of an episode
    assert "Australasian Triage Scale" in early[0]["content"]
    assert "[Vital] sbp = 98 mmHg" in later[1]["content"]
    assert "- vital signs: sbp 98 mmHg (not yet taken: temperature, heartrate, resprate, o2sat)" in later[1]["content"]
    assert '"check_vital"' in early[1]["content"] and '"check_vital"' not in later[1]["content"]


def test_patient_prompt_handles_non_numeric_pain(patient_persona, case):
    numeric = prompts.patient_messages(patient_persona, case, None, [])[0]["content"]
    unknown = prompts.patient_messages(patient_persona, case.model_copy(update={"pain": "UA"}), None, [])[0]["content"]
    assert "approximately 7 out of 10" in numeric
    assert "out of 10" not in unknown
    assert "verbosity: terse (answers in a few words" in numeric


def test_ethnicity_and_tts_instruction_are_hidden_from_models(nurse_persona, patient_persona, case):
    nurse = nurse_persona.model_copy(update={"instruction": "Speak with an Australian accent."})
    system = prompts.nurse_messages(nurse, "esi", "- gender: male", [], ("utterance", "end"), ())[0]["content"]
    patient = prompts.patient_messages(patient_persona, case, None, [])[0]["content"]
    for text in (system, patient):
        assert "ethnicity" not in text and "Australian" not in text and "accent" not in text
    assert "expertise: expert (asks fewer, targeted" in system


def test_vital_units(case):
    history = [{"turn": 0, "actor": "system", "event": "vital", "name": n, "value": v}
               for n, v in [("temperature", 98.6), ("temperature", 37.2), ("o2sat", 91.0), ("resprate", 26.0)]]
    text = prompts.transcript(history)
    assert "98.6 °F" in text and "37.2 °C" in text and "o2sat = 91%" in text and "26 breaths/min" in text


def test_nurse_sees_the_patient_at_the_desk(case, patient_persona):
    arrived = case.model_copy(update={"gender": "male", "arrival_transport": "AMBULANCE"})
    view = prompts.patient_view(arrived, patient_persona)
    assert view == "- gender: male\n- age group: adult\n- arrived by: ambulance"
    assert "arrived by" not in prompts.patient_view(case.model_copy(update={"arrival_transport": "UNKNOWN"}), patient_persona)


def test_patient_knows_how_unwell_but_not_the_level_name(case, patient_persona):
    system = prompts.patient_messages(patient_persona, case, None, [])[0]["content"]
    assert prompts.SEVERITY[2] in system and "Never mention triage levels" in system


def test_script_prompt_uses_case_facts_and_persona(case, patient_persona):
    rich = case.model_copy(update={"gender": "female", "acuity_scale": "esi", "diagnoses": ["Syncope and collapse"],
                                   "medications": ["furosemide"], "arrival_transport": "WALK IN", "specialisation": "CAR"})
    user = prompts.script_messages(rich, patient_persona)[1]["content"]
    for fact in ("Syncope", "female", "walk in", "112 bpm", "ESI 2 of 5", "CAR", "Syncope and collapse",
                 "home medications (the patient takes these): furosemide", "PATIENT PERSONA", "recall reliability: reliable"):
        assert fact in user
    assert "ethnicity" not in user


def test_in_situ_check_sees_only_what_is_visible_on_arrival(case):
    rich = case.model_copy(update={"arrival_transport": "AMBULANCE", "acuity_scale": "esi",
                                   "diagnoses": ["Syncope and collapse"], "medications": ["furosemide"],
                                   "prior_ed_visits": 3})
    user = prompts.in_situ_messages(rich)[1]["content"]
    assert "arrived by: ambulance" in user and "Syncope" in user and "112 bpm" in user
    for hidden in ("acuity", "ESI", "2 of 5", "collapse", "furosemide", "visits"):
        assert hidden not in user


def test_nurse_persona_shapes_questions_not_the_triage_level(nurse_persona):
    system = prompts.nurse_messages(nurse_persona, "esi", "- gender: male", [], ("utterance", "end"), ())[0]["content"]
    assert "shapes which questions you ask" in system and "does not change how you apply the triage algorithm" in system


def test_judge_checks_are_phrased_so_true_means_pass(nurse_persona):
    system = prompts.judge_messages("nurse", "Hi", nurse_persona, [], patient="- gender: male")[0]["content"]
    assert "true if the utterance passes it" in system
    assert "never" not in system.lower()  # negated criteria get inverted by small models


def test_prior_ed_visits_reach_patient_and_master(case, patient_persona):
    frequent = case.model_copy(update={"prior_ed_visits": 5})
    assert "this emergency department in the past year, before today: 5" in prompts.patient_messages(patient_persona, frequent, None, [])[0]["content"]
    assert "ED visits in the past year, before this one: 5" in prompts.script_messages(frequent, patient_persona)[1]["content"]
    assert "past year" not in prompts.patient_messages(patient_persona, case, None, [])[0]["content"]  # unknown: not shown


def test_patients_ask_instead_of_bluffing_and_the_master_ignores_non_answers(patient_persona, case):
    system = prompts.patient_messages(patient_persona, case, None, [])[0]["content"]
    assert "ask what they mean" in system and "never use medical terms yourself" in system
    reading = prompts.reading_messages({}, "Any palpitations?", "I'm just scared.")[0]["content"]
    assert "adds nothing about it: never record a denial" in " ".join(reading.split())
