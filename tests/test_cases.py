import json

from triagesim.cases import Case, load_cases


def test_legacy_ground_truth_dict_validates():
    case = Case.model_validate(
        {"case_id": 1, "chiefcomplaint": "Syncope", "vitals": {"heartrate": 112}, "acuity": 2, "pain": 7}
    )
    assert (case.case_id, case.chief_complaint, case.pain) == ("1", "Syncope", "7")


def test_jsonl_with_nested_vitals(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [{"case_id": "a", "chief_complaint": "Chest pain", "vitals": {"sbp": 140, "o2sat": None}, "acuity": 2}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    [case] = load_cases(path)
    assert case.vitals == {"sbp": 140.0, "o2sat": None}


def test_mimic_style_csv(tmp_path):
    path = tmp_path / "triage.csv"
    path.write_text(
        "subject_id,stay_id,temperature,heartrate,resprate,o2sat,sbp,dbp,pain,acuity,chiefcomplaint\n"
        "10,30001,98.6,88,16,,120,80,UA,3,Abd pain\n"
        "11,30002,,,,,,,8,,Headache\n"
    )
    first, second = load_cases(path)
    assert first.case_id == "30001"
    assert first.vitals == {"temperature": 98.6, "heartrate": 88, "resprate": 16, "o2sat": None, "sbp": 120}
    assert (first.pain, first.acuity) == ("UA", 3)
    assert second.acuity is None and set(second.vitals.values()) == {None}


def test_json_list_is_numbered_without_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"chief_complaint": "Rash"}, {"chief_complaint": "Cough"}]))
    assert [c.case_id for c in load_cases(path)] == ["0", "1"]


def test_extra_fields_gender_and_scale(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"stay_id": 7, "chiefcomplaint": "Fall", "gender": "F", "heartrate": 90,
                                "arrival_transport": "AMBULANCE", "acuity_scale": "ats"}) + "\n")
    [case] = load_cases(path)
    assert (case.case_id, case.gender, case.acuity_scale) == ("7", "female", "ats")
    assert case.model_extra == {"arrival_transport": "AMBULANCE"}  # flat vitals are not duplicated
    assert case.model_dump()["arrival_transport"] == "AMBULANCE"
