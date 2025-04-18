import os
from typing import Optional, List, Dict, Union, Tuple

import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime
from datetime import datetime
from pyhealth.datasets.utils import strptime, padyear
from tqdm import tqdm
# TODO: add other tables


class MIMIC4Dataset(BaseEHRDataset):
    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.

    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICD10PROC
            code) for patients.
        - prescriptions: contains medication related order entries (NDC code)
            for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code)
            for patients

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> dataset = MIMIC4Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...         tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["admittime"].values[0]),
                    discharge_time=strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        print("Bug this start here")
        df_group = df_group.apply(  # fixme
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        print("Bug end")
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - diagnosis_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in diagnoses_icd
                table, so we set it to None.
        """
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in procedures_icd
                table, so we set it to None.
        """
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table.

        Will be called in `self.parse_tables()`

        Docs:
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses labevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["charttime"], v_info["itemid"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_hcpcsevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses hcpcsevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - hcpcsevents: https://mimic.mit.edu/docs/iv/modules/hosp/hcpcsevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in hcpcsevents
                table, so we set it to None.
        """
        table = "hcpcsevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "hcpcs_cd": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "hcpcs_cd"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of hcpcsevents (per patient)
        def hcpcsevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["hcpcs_cd"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_HCPCS_CD",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: hcpcsevents_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients


class eICUDataset(BaseEHRDataset):
    """Base dataset for eICU dataset.

    The eICU dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://eicu-crd.mit.edu/.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and a ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
    hospital admission can have multiple ICU stays. The data in eICU is centered
    around the ICU stay and all timestamps are relative to the ICU admission time.
    Thus, we only know the order of ICU stays within a hospital admission, but not
    the order of hospital admissions within a patient. As a result, we use `Patient`
    object to represent a hospital admission of a patient, and use `Visit` object to
    store the ICU stays within that hospital admission.

    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            and diagnosis information (under attr_dict) for patients
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code)
            for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH)
            conducted for patients.
        - admissionDx:  table contains the primary diagnosis for admission to
            the ICU per the APACHE scoring criteria. (eICU_ADMITDXPATH)

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> dataset = eICUDataset(
        ...         root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...         tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(self, **kwargs):
        # store a mapping from visit_id to patient_id
        # will be used to parse clinical tables as they only contain visit_id
        self.visit_id_to_patient_id: Dict[str, str] = {}
        self.visit_id_to_encounter_time: Dict[str, datetime] = {}
        super(eICUDataset, self).__init__(**kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patient and hospital tables.

        Will be called in `self.parse_tables()`.

        Docs:
            - patient: https://eicu-crd.mit.edu/eicutables/patient/
            - hospital: https://eicu-crd.mit.edu/eicutables/hospital/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            We use `Patient` object to represent a hospital admission of a patient,
            and use `Visit` object to store the ICU stays within that hospital
            admission.
        """
        # read patient table
        patient_df = pd.read_csv(
            os.path.join(self.root, "patient.csv"),
            dtype={
                "uniquepid": str,
                "patienthealthsystemstayid": str,
                "patientunitstayid": str,
            },
            nrows=5000 if self.dev else None,
        )
        # read hospital table
        hospital_df = pd.read_csv(os.path.join(self.root, "hospital.csv"))
        hospital_df.region = hospital_df.region.fillna("Unknown").astype(str)
        # merge patient and hospital tables
        df = pd.merge(patient_df, hospital_df, on="hospitalid", how="left")
        # sort by ICU admission and discharge time
        df["neg_hospitaladmitoffset"] = -df["hospitaladmitoffset"]
        df = df.sort_values(
            [
                "uniquepid",
                "patienthealthsystemstayid",
                "neg_hospitaladmitoffset",
                "unitdischargeoffset",
            ],
            ascending=True,
        )
        # group by patient and hospital admission
        df_group = df.groupby(["uniquepid", "patienthealthsystemstayid"])
        # load patients
        for (p_id, ha_id), p_info in tqdm(df_group, desc="Parsing patients"):
            # each Patient object is a single hospital admission of a patient
            patient_id = f"{p_id}+{ha_id}"

            # hospital admission time (Jan 1 of hospitaldischargeyear, 00:00:00)
            ha_datetime = strptime(padyear(str(p_info["hospitaldischargeyear"].values[0])))

            # no exact birth datetime in eICU
            # use hospital admission time and age to approximate birth datetime
            age = p_info["age"].values[0]
            if pd.isna(age):
                birth_datetime = None
            elif age == "> 89":
                birth_datetime = ha_datetime - pd.DateOffset(years=89)
            else:
                birth_datetime = ha_datetime - pd.DateOffset(years=int(age))

            # no exact death datetime in eICU
            # use hospital discharge time to approximate death datetime
            death_datetime = None
            if p_info["hospitaldischargestatus"].values[0] == "Expired":
                ha_los_min = (
                    p_info["hospitaldischargeoffset"].values[0]
                    - p_info["hospitaladmitoffset"].values[0]
                )
                death_datetime = ha_datetime + pd.Timedelta(minutes=ha_los_min)

            patient = Patient(
                patient_id=patient_id,
                birth_datetime=birth_datetime,
                death_datetime=death_datetime,
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0],
            )

            # load visits
            for v_id, v_info in p_info.groupby("patientunitstayid"):
                # each Visit object is a single ICU stay within a hospital admission

                # base time is the hospital admission time
                unit_admit = v_info["neg_hospitaladmitoffset"].values[0]
                unit_discharge = unit_admit + v_info["unitdischargeoffset"].values[0]
                encounter_time = ha_datetime + pd.Timedelta(minutes=unit_admit)
                discharge_time = ha_datetime + pd.Timedelta(minutes=unit_discharge)

                visit = Visit(
                    visit_id=v_id,
                    patient_id=patient_id,
                    encounter_time=encounter_time,
                    discharge_time=discharge_time,
                    discharge_status=v_info["unitdischargestatus"].values[0],
                    hospital_id=v_info["hospitalid"].values[0],
                    region=v_info["region"].values[0],
                )

                # add visit
                patient.add_visit(visit)
                # add visit id to patient id mapping
                self.visit_id_to_patient_id[v_id] = patient_id
                # add visit id to encounter time mapping
                self.visit_id_to_encounter_time[v_id] = encounter_time
            # add patient
            patients[patient_id] = patient
        return patients

    def parse_diagnosis(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis table.

        Will be called in `self.parse_tables()`.

        Docs:
            - diagnosis: https://eicu-crd.mit.edu/eicutables/diagnosis/

        Args:
            patients: a dict of Patient objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            This table contains both ICD9CM and ICD10CM codes in one single
                cell. We need to use medcode to distinguish them.
        """

        # load ICD9CM and ICD10CM coding systems
        from pyhealth.medcode import ICD9CM, ICD10CM

        icd9cm = ICD9CM()
        icd10cm = ICD10CM()

        def icd9cm_or_icd10cm(code):
            if code in icd9cm:
                return "ICD9CM"
            elif code in icd10cm:
                return "ICD10CM"
            else:
                return "Unknown"

        table = "diagnosis"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "icd9code": str, "diagnosisstring": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "icd9code", "diagnosisstring"])
        # sort by diagnosisoffset
        df = df.sort_values(["patientunitstayid", "diagnosisoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of diagnosis (per visit)
        def diagnosis_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, codes, dxstr in zip(v_info["diagnosisoffset"], v_info["icd9code"],
                                            v_info["diagnosisstring"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                codes = [c.strip() for c in codes.split(",")]
                # for each code in a single cell (mixed ICD9CM and ICD10CM)
                for code in codes:
                    vocab = icd9cm_or_icd10cm(code)
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=vocab,
                        visit_id=v_id,
                        patient_id=patient_id,
                        timestamp=timestamp,
                        diagnosisString=dxstr
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(lambda x: diagnosis_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_treatment(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses treatment table.

        Will be called in `self.parse_tables()`.

        Docs:
            - treatment: https://eicu-crd.mit.edu/eicutables/treatment/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "treatment"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "treatmentstring": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "treatmentstring"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "treatmentoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of treatment (per visit)
        def treatment_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(
                v_info["treatmentoffset"], v_info["treatmentstring"]
            ):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_TREATMENTSTRING",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)

            return events

        # parallel apply
        group_df = group_df.apply(lambda x: treatment_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_medication(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses medication table.

        Will be called in `self.parse_tables()`.

        Docs:
            - medication: https://eicu-crd.mit.edu/eicutables/medication/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "medication"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"patientunitstayid": str, "drugname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "drugname"])
        # sort by drugstartoffset
        df = df.sort_values(["patientunitstayid", "drugstartoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of medication (per visit)
        def medication_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(v_info["drugstartoffset"], v_info["drugname"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_DRUGNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(lambda x: medication_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_lab(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses lab table.

        Will be called in `self.parse_tables()`.

        Docs:
            - lab: https://eicu-crd.mit.edu/eicutables/lab/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "lab"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "labname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "labname"])
        # sort by labresultoffset
        df = df.sort_values(["patientunitstayid", "labresultoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of lab (per visit)
        def lab_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(v_info["labresultoffset"], v_info["labname"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_LABNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(lambda x: lab_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_physicalexam(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses physicalExam table.

        Will be called in `self.parse_tables()`.

        Docs:
            - physicalExam: https://eicu-crd.mit.edu/eicutables/physicalexam/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "physicalExam"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "physicalexampath": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "physicalexampath"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "physicalexamoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of physicalExam (per visit)
        def physicalExam_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(
                v_info["physicalexamoffset"], v_info["physicalexampath"]
            ):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_PHYSICALEXAMPATH",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(lambda x: physicalExam_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_admissiondx(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses admissionDx (admission diagnosis) table.

        Will be called in `self.parse_tables()`.

        Docs:
            - admissionDx: https://eicu-crd.mit.edu/eicutables/admissiondx/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "admissionDx"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "admitdxpath": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "admitdxpath"])
        # sort by admitDxEnteredOffset
        df = df.sort_values(["patientunitstayid", "admitdxenteredoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of admissionDx (per visit)
        def admissionDx_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(
                v_info["admitdxenteredoffset"], v_info["admitdxpath"]
            ):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_ADMITDXPATH",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(lambda x: admissionDx_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


class MIMIC3Dataset(BaseEHRDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NDC code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code)
            for patients

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...         tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the mimic-3 table result.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        table = "DIAGNOSES_ICD"
        self.code_vocs["conditions"] = "ICD9CM"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        self.code_vocs["procedures"] = "ICD9PROC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "NDC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        self.code_vocs["labs"] = "MIMIC3_ITEMID"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["CHARTTIME"], v_info["ITEMID"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.apply(
            lambda x: lab_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients



if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents", "hcpcsevents"],
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.info()
