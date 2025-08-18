FEW_SHOT_WITH_MULTI_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an advanced assistant trained to process biomedical text for Named Entity Recognition (NER) and Relation Extraction (RE). "
            "Your task is to analyze user-provided text, identify all unique and contextually relevant entities, and infer directed relationships "
            "between these entities based on the context. Ensure that all relations exist only between annotated entities. "
            "Entities and relationships should be human-readable and natural, reflecting real-world concepts and connections. "
            "Output the annotated data in JSON format, structured as follows:\n\n"
            """{"entities": [{"id": 0, "text": "ner_string_0", "type": "ner_type_string_0"}, {"id": 1, "text": "ner_string_1", "type": "ner_type_string_1"}], "relations": [{"head": 0, "tail": 1, "type": "re_type_string_0"}]}"""
            "\n\nEnsure that the output captures all significant entities and their directed relationships in a clear and concise manner."
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "Olmesartan medoxomil tablet contains olmesartan, a drug that inhibits the renin-angiotensin system (RAS). Drugs that inhibit the RAS can cause hyperkalemia. Monitor serum electrolytes periodically." '
            "Analyze this text, identify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Olmesartan medoxomil tablet", "type": "Drug product"}, {"id": 1, "text": "olmesartan", "type": "Drug"}, {"id": 2, "text": "renin-angiotensin system (RAS)", "type": "Biological system"}, {"id": 3, "text": "hyperkalemia", "type": "Adverse drug event"}, {"id": 4, "text": "serum electrolytes", "type": "Biomarker"}, {"id": 5, "text": "periodically", "type": "Time frequency"}], "relations": [{"head": 0, "tail": 1, "type": "contains"}, {"head": 1, "tail": 2, "type": "inhibits"}, {"head": 1, "tail": 3, "type": "can cause"}, {"head": 4, "tail": 5, "type": "Monitor"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "This study was aimed to evaluate the effects of different Orthokeratology on myopia control and visual quality. The different optical zone of Orthokeratology lens was divided into 4 groups, ranged from 5.5 mm to 6 mm. And the control group subjects with the single glasses was included. The effectiveness of Orthokeratology was measured by axial length progression. The visual quality of subjects was evaluated by a questionnaire, contrast sensitivity and wavefront aberration." '
            "Analyze this text, identify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Orthokeratology", "type": "Medical intervention"}, {"id": 1, "text": "myopia", "type": "Condition"}, {"id": 2, "text": "visual quality", "type": "Outcome"}, {"id": 3, "text": "optical zone", "type": "Lens feature"}, {"id": 4, "text": "5.5 mm to 6 mm", "type": "Size range"}, {"id": 5, "text": "control group", "type": "Study group"}, {"id": 6, "text": "single glasses", "type": "Device"}, {"id": 7, "text": "axial length progression", "type": "Study metric"}, {"id": 8, "text": "questionnaire", "type": "Evaluation tool"}, {"id": 9, "text": "contrast sensitivity", "type": "Evaluation tool"}, {"id": 10, "text": "wavefront aberration", "type": "Evaluation tool"}], "relations": [{"head": 0, "tail": 1, "type": "evaluates effect on"}, {"head": 0, "tail": 2, "type": "evaluates effect on"}, {"head": 0, "tail": 3, "type": "features"}, {"head": 3, "tail": 4, "type": "has size range"}, {"head": 5, "tail": 6, "type": "includes"}, {"head": 1, "tail": 7, "type": "measured by"}, {"head": 2, "tail": 8, "type": "evaluated using"}, {"head": 2, "tail": 9, "type": "evaluated using"}, {"head": 2, "tail": 10, "type": "evaluated using"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "Subjects will receive a 100mL dose of IV saline every 6 hours for 24 hours. The first dose will be administered prior to anesthesia induction, approximately 30 minutes before skin incision. A total of 4 doses will be given." '
            "Analyze this text, identify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "IV", "type": "Administration route"}, {"id": 1, "text": "saline", "type": "Drug"}, {"id": 2, "text": "100mL", "type": "Dosage"}, {"id": 3, "text": "every 6 hours", "type": "Frequency"}, {"id": 4, "text": "24 hours", "type": "Duration"}, {"id": 5, "text": "4 doses", "type": "Quantity"}, {"id": 6, "text": "anesthesia induction", "type": "Procedure"}, {"id": 7, "text": "30 minutes before skin incision", "type": "Timing"}], "relations": [{"head": 1, "tail": 0, "type": "route of"}, {"head": 1, "tail": 2, "type": "administered as"}, {"head": 1, "tail": 3, "type": "administered every"}, {"head": 1, "tail": 4, "type": "administered for"}, {"head": 1, "tail": 5, "type": "total quantity"}, {"head": 1, "tail": 7, "type": "first dose before"}, {"head": 1, "tail": 6, "type": "related to procedure"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "BACKGROUND: Plantar warts, or verruca plantaris, are common cutaneous lesions of the plantar surface caused by the human papilloma virus (HPV) infection. Current treatments include salicylic acid, cryotherapy, bleomycin, and immunotherapy; however, they have limitations of low-cure rates or side effects. Plantar warts treated with bee venom (BV) have not been reported. Here we present a case of a patient who showed complete remission of multiple plantar warts after BV treatment.\nCASE PRESENTATION: A 13-year-old boy experienced total of 16 plantar warts and visited department of dermatology of Korean Medicine. BV was injected into wart sites. Warts were completely removed at the end of the 8-week treatment with no adverse event recorded. There was improvement during the follow-up after 4 weeks.\nCONCLUSION: This is the first case of plantar warts that improved with BV injection. This study suggests that BV may be a therapeutic option for individuals who cannot receive cryotherapy due to pain during and after treatment or who have refractory or relapsed warts." '
            "Analyze this text, identify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Plantar warts", "type": "Condition"}, {"id": 1, "text": "verruca plantaris", "type": "Synonym"}, {"id": 2, "text": "cutaneous lesions", "type": "Condition"}, {"id": 3, "text": "plantar surface", "type": "Location"}, {"id": 4, "text": "human papilloma virus (HPV)", "type": "Pathogen"}, {"id": 5, "text": "salicylic acid", "type": "Treatment"}, {"id": 6, "text": "cryotherapy", "type": "Treatment"}, {"id": 7, "text": "bleomycin", "type": "Treatment"}, {"id": 8, "text": "immunotherapy", "type": "Treatment"}, {"id": 9, "text": "low-cure rates", "type": "Limitation"}, {"id": 10, "text": "side effects", "type": "Limitation"}, {"id": 11, "text": "bee venom (BV)", "type": "Treatment"}, {"id": 12, "text": "complete remission", "type": "Outcome"}, {"id": 13, "text": "8-week treatment", "type": "Duration"}, {"id": 14, "text": "16 plantar warts", "type": "Condition"}, {"id": 15, "text": "wart sites", "type": "Location"}, {"id": 16, "text": "no adverse event", "type": "Outcome"}, {"id": 17, "text": "follow-up after 4 weeks", "type": "Event"}, {"id": 18, "text": "first case", "type": "Event"}, {"id": 19, "text": "refractory or relapsed warts", "type": "Condition"}, {"id": 20, "text": "pain during and after treatment", "type": "Limitation"}], "relations": [{"head": 0, "tail": 1, "type": "synonym of"}, {"head": 0, "tail": 2, "type": "type of"}, {"head": 0, "tail": 4, "type": "caused by"}, {"head": 0, "tail": 3, "type": "occurs in"}, {"head": 5, "tail": 9, "type": "associated with"}, {"head": 5, "tail": 10, "type": "associated with"}, {"head": 6, "tail": 9, "type": "associated with"}, {"head": 6, "tail": 10, "type": "associated with"}, {"head": 11, "tail": 0, "type": "considered for"}, {"head": 11, "tail": 12, "type": "resulted in"}, {"head": 14, "tail": 11, "type": "treated with"}, {"head": 11, "tail": 15, "type": "administered to"}, {"head": 13, "tail": 16, "type": "resulted in"}, {"head": 18, "tail": 0, "type": "involved"}, {"head": 19, "tail": 11, "type": "treated with"}, {"head": 6, "tail": 20, "type": "associated with"}]}"""
        ),
    },
]

FEW_SHOT_WITH_SINGLE_SYSTEM_MESSAGE = [
    {
        "role": "system",
        "content": (
            "You are an advanced assistant trained to process biomedical text for Named Entity Recognition (NER) and Relation Extraction (RE). "
            "Your task is to analyze user-provided text, identify all unique and contextually relevant entities, and infer directed relationships "
            "between these entities based on the context. Ensure that all relations exist only between annotated entities. "
            "Entities and relationships should be human-readable and natural, reflecting real-world concepts and connections. "
            "Output the annotated data in JSON format, structured as follows:\n\n"
            """{"entities": [{"id": 0, "text": "ner_string_0", "type": "ner_type_string_0"}, {"id": 1, "text": "ner_string_1", "type": "ner_type_string_1"}], "relations": [{"head": 0, "tail": 1, "type": "re_type_string_0"}]}"""
            "\n\nEnsure that the output captures all significant entities and their directed relationships in a clear and concise manner."
            "Below are examples that demonstrates how to make NER and RE:"
            'Here is a text input: "Olmesartan medoxomil tablet contains olmesartan, a drug that inhibits the renin-angiotensin system (RAS). Drugs that inhibit the RAS can cause hyperkalemia. Monitor serum electrolytes periodically." '
            "Output is:\n"
            """{"entities": [{"id": 0, "text": "Olmesartan medoxomil tablet", "type": "Drug product"}, {"id": 1, "text": "olmesartan", "type": "Drug"}, {"id": 2, "text": "renin-angiotensin system (RAS)", "type": "Biological system"}, {"id": 3, "text": "hyperkalemia", "type": "Adverse drug event"}, {"id": 4, "text": "serum electrolytes", "type": "Biomarker"}, {"id": 5, "text": "periodically", "type": "Time frequency"}], "relations": [{"head": 0, "tail": 1, "type": "contains"}, {"head": 1, "tail": 2, "type": "inhibits"}, {"head": 1, "tail": 3, "type": "can cause"}, {"head": 4, "tail": 5, "type": "Monitor"}]}"""
            'Here is a text input: "This study was aimed to evaluate the effects of different Orthokeratology on myopia control and visual quality. The different optical zone of Orthokeratology lens was divided into 4 groups, ranged from 5.5 mm to 6 mm. And the control group subjects with the single glasses was included. The effectiveness of Orthokeratology was measured by axial length progression. The visual quality of subjects was evaluated by a questionnaire, contrast sensitivity and wavefront aberration." '
            "Output is:\n"
            """{"entities": [{"id": 0, "text": "Orthokeratology", "type": "Medical intervention"}, {"id": 1, "text": "myopia", "type": "Condition"}, {"id": 2, "text": "visual quality", "type": "Outcome"}, {"id": 3, "text": "optical zone", "type": "Lens feature"}, {"id": 4, "text": "5.5 mm to 6 mm", "type": "Size range"}, {"id": 5, "text": "control group", "type": "Study group"}, {"id": 6, "text": "single glasses", "type": "Device"}, {"id": 7, "text": "axial length progression", "type": "Study metric"}, {"id": 8, "text": "questionnaire", "type": "Evaluation tool"}, {"id": 9, "text": "contrast sensitivity", "type": "Evaluation tool"}, {"id": 10, "text": "wavefront aberration", "type": "Evaluation tool"}], "relations": [{"head": 0, "tail": 1, "type": "evaluates effect on"}, {"head": 0, "tail": 2, "type": "evaluates effect on"}, {"head": 0, "tail": 3, "type": "features"}, {"head": 3, "tail": 4, "type": "has size range"}, {"head": 5, "tail": 6, "type": "includes"}, {"head": 1, "tail": 7, "type": "measured by"}, {"head": 2, "tail": 8, "type": "evaluated using"}, {"head": 2, "tail": 9, "type": "evaluated using"}, {"head": 2, "tail": 10, "type": "evaluated using"}]}"""
            'Here is a text input: "Subjects will receive a 100mL dose of IV saline every 6 hours for 24 hours. The first dose will be administered prior to anesthesia induction, approximately 30 minutes before skin incision. A total of 4 doses will be given." '
            "Output is:\n"
            """{"entities": [{"id": 0, "text": "IV", "type": "Administration route"}, {"id": 1, "text": "saline", "type": "Drug"}, {"id": 2, "text": "100mL", "type": "Dosage"}, {"id": 3, "text": "every 6 hours", "type": "Frequency"}, {"id": 4, "text": "24 hours", "type": "Duration"}, {"id": 5, "text": "4 doses", "type": "Quantity"}, {"id": 6, "text": "anesthesia induction", "type": "Procedure"}, {"id": 7, "text": "30 minutes before skin incision", "type": "Timing"}], "relations": [{"head": 1, "tail": 0, "type": "route of"}, {"head": 1, "tail": 2, "type": "administered as"}, {"head": 1, "tail": 3, "type": "administered every"}, {"head": 1, "tail": 4, "type": "administered for"}, {"head": 1, "tail": 5, "type": "total quantity"}, {"head": 1, "tail": 7, "type": "first dose before"}, {"head": 1, "tail": 6, "type": "related to procedure"}]}"""
            'Here is a text input: "BACKGROUND: Plantar warts, or verruca plantaris, are common cutaneous lesions of the plantar surface caused by the human papilloma virus (HPV) infection. Current treatments include salicylic acid, cryotherapy, bleomycin, and immunotherapy; however, they have limitations of low-cure rates or side effects. Plantar warts treated with bee venom (BV) have not been reported. Here we present a case of a patient who showed complete remission of multiple plantar warts after BV treatment.\nCASE PRESENTATION: A 13-year-old boy experienced total of 16 plantar warts and visited department of dermatology of Korean Medicine. BV was injected into wart sites. Warts were completely removed at the end of the 8-week treatment with no adverse event recorded. There was improvement during the follow-up after 4 weeks.\nCONCLUSION: This is the first case of plantar warts that improved with BV injection. This study suggests that BV may be a therapeutic option for individuals who cannot receive cryotherapy due to pain during and after treatment or who have refractory or relapsed warts." '
            "Output is:\n"
            """{"entities": [{"id": 0, "text": "Plantar warts", "type": "Condition"}, {"id": 1, "text": "verruca plantaris", "type": "Synonym"}, {"id": 2, "text": "cutaneous lesions", "type": "Condition"}, {"id": 3, "text": "plantar surface", "type": "Location"}, {"id": 4, "text": "human papilloma virus (HPV)", "type": "Pathogen"}, {"id": 5, "text": "salicylic acid", "type": "Treatment"}, {"id": 6, "text": "cryotherapy", "type": "Treatment"}, {"id": 7, "text": "bleomycin", "type": "Treatment"}, {"id": 8, "text": "immunotherapy", "type": "Treatment"}, {"id": 9, "text": "low-cure rates", "type": "Limitation"}, {"id": 10, "text": "side effects", "type": "Limitation"}, {"id": 11, "text": "bee venom (BV)", "type": "Treatment"}, {"id": 12, "text": "complete remission", "type": "Outcome"}, {"id": 13, "text": "8-week treatment", "type": "Duration"}, {"id": 14, "text": "16 plantar warts", "type": "Condition"}, {"id": 15, "text": "wart sites", "type": "Location"}, {"id": 16, "text": "no adverse event", "type": "Outcome"}, {"id": 17, "text": "follow-up after 4 weeks", "type": "Event"}, {"id": 18, "text": "first case", "type": "Event"}, {"id": 19, "text": "refractory or relapsed warts", "type": "Condition"}, {"id": 20, "text": "pain during and after treatment", "type": "Limitation"}], "relations": [{"head": 0, "tail": 1, "type": "synonym of"}, {"head": 0, "tail": 2, "type": "type of"}, {"head": 0, "tail": 4, "type": "caused by"}, {"head": 0, "tail": 3, "type": "occurs in"}, {"head": 5, "tail": 9, "type": "associated with"}, {"head": 5, "tail": 10, "type": "associated with"}, {"head": 6, "tail": 9, "type": "associated with"}, {"head": 6, "tail": 10, "type": "associated with"}, {"head": 11, "tail": 0, "type": "considered for"}, {"head": 11, "tail": 12, "type": "resulted in"}, {"head": 14, "tail": 11, "type": "treated with"}, {"head": 11, "tail": 15, "type": "administered to"}, {"head": 13, "tail": 16, "type": "resulted in"}, {"head": 18, "tail": 0, "type": "involved"}, {"head": 19, "tail": 11, "type": "treated with"}, {"head": 6, "tail": 20, "type": "associated with"}]}"""

        ),
    }
]

FEW_SHOT_WITH_MULTI_MESSAGES_AND_ENTITIES = [
    {
        "role": "system",
        "content": (
            "You are an advanced assistant trained to process biomedical text for Named Entity Recognition (NER) and Relation Extraction (RE). "
            "Your task is to analyze user-provided text and provided entities selecting all unique and contextually relevant entities, and infer directed relationships. "
            "between these entities based on the context. Ensure that all relations exist only between annotated entities. "
            "Entities and relationships should be human-readable and natural, reflecting real-world concepts and connections. "
            "Output the annotated data in JSON format, structured as follows:\n\n"
            """{"entities": [{"id": 0, "text": "ner_string_0", "type": "ner_type_string_0"}, {"id": 1, "text": "ner_string_1", "type": "ner_type_string_1"}], "relations": [{"head": 0, "tail": 1, "type": "re_type_string_0"}]}"""
            "\n\nEnsure that the output captures all significant entities and their directed relationships in a clear and concise manner."
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "Olmesartan medoxomil tablet contains olmesartan, a drug that inhibits the renin-angiotensin system (RAS). Drugs that inhibit the RAS can cause hyperkalemia. Monitor serum electrolytes periodically." '
            """Here is the list of input entities: ['Olmesartan medoxomil tablet', 'olmesartan', 'drug', 'renin - angiotensin system', 'RAS', 'Drugs', 'RAS', 'hyperkalemia', 'serum electrolytes']"""
            "Analyze this text, select and classify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Olmesartan medoxomil tablet", "type": "Drug product"}, {"id": 1, "text": "olmesartan", "type": "Drug"}, {"id": 2, "text": "renin-angiotensin system (RAS)", "type": "Biological system"}, {"id": 3, "text": "hyperkalemia", "type": "Adverse drug event"}, {"id": 4, "text": "serum electrolytes", "type": "Biomarker"}, {"id": 5, "text": "periodically", "type": "Time frequency"}], "relations": [{"head": 0, "tail": 1, "type": "contains"}, {"head": 1, "tail": 2, "type": "inhibits"}, {"head": 1, "tail": 3, "type": "can cause"}, {"head": 4, "tail": 5, "type": "Monitor"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "This study was aimed to evaluate the effects of different Orthokeratology on myopia control and visual quality. The different optical zone of Orthokeratology lens was divided into 4 groups, ranged from 5.5 mm to 6 mm. And the control group subjects with the single glasses was included. The effectiveness of Orthokeratology was measured by axial length progression. The visual quality of subjects was evaluated by a questionnaire, contrast sensitivity and wavefront aberration." '
            """Here is the list of input entities: ['study', 'effects', 'different Orthokeratology', 'myopia control', 'visual quality', 'different optical zone', 'Orthokeratology lens', '4 groups', '5.5 mm', '6 mm', 'control group subjects', 'single glasses', 'effectiveness', 'Orthokeratology', 'axial length progression', 'visual quality', 'subjects', 'questionnaire', 'contrast sensitivity', 'wavefront aberration']"""
            "Analyze this text, select and classify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Orthokeratology", "type": "Medical intervention"}, {"id": 1, "text": "myopia", "type": "Condition"}, {"id": 2, "text": "visual quality", "type": "Outcome"}, {"id": 3, "text": "optical zone", "type": "Lens feature"}, {"id": 4, "text": "5.5 mm to 6 mm", "type": "Size range"}, {"id": 5, "text": "control group", "type": "Study group"}, {"id": 6, "text": "single glasses", "type": "Device"}, {"id": 7, "text": "axial length progression", "type": "Study metric"}, {"id": 8, "text": "questionnaire", "type": "Evaluation tool"}, {"id": 9, "text": "contrast sensitivity", "type": "Evaluation tool"}, {"id": 10, "text": "wavefront aberration", "type": "Evaluation tool"}], "relations": [{"head": 0, "tail": 1, "type": "evaluates effect on"}, {"head": 0, "tail": 2, "type": "evaluates effect on"}, {"head": 0, "tail": 3, "type": "features"}, {"head": 3, "tail": 4, "type": "has size range"}, {"head": 5, "tail": 6, "type": "includes"}, {"head": 1, "tail": 7, "type": "measured by"}, {"head": 2, "tail": 8, "type": "evaluated using"}, {"head": 2, "tail": 9, "type": "evaluated using"}, {"head": 2, "tail": 10, "type": "evaluated using"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "Subjects will receive a 100mL dose of IV saline every 6 hours for 24 hours. The first dose will be administered prior to anesthesia induction, approximately 30 minutes before skin incision. A total of 4 doses will be given." '
            """Here is the list of input entities: ['Subjects', '100mL dose', 'IV saline', '24 hours', 'first dose', 'anesthesia induction', 'skin incision', 'total', '4 doses']"""
            "Analyze this text, select and classify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "IV", "type": "Administration route"}, {"id": 1, "text": "saline", "type": "Drug"}, {"id": 2, "text": "100mL", "type": "Dosage"}, {"id": 3, "text": "every 6 hours", "type": "Frequency"}, {"id": 4, "text": "24 hours", "type": "Duration"}, {"id": 5, "text": "4 doses", "type": "Quantity"}, {"id": 6, "text": "anesthesia induction", "type": "Procedure"}, {"id": 7, "text": "30 minutes before skin incision", "type": "Timing"}], "relations": [{"head": 1, "tail": 0, "type": "route of"}, {"head": 1, "tail": 2, "type": "administered as"}, {"head": 1, "tail": 3, "type": "administered every"}, {"head": 1, "tail": 4, "type": "administered for"}, {"head": 1, "tail": 5, "type": "total quantity"}, {"head": 1, "tail": 7, "type": "first dose before"}, {"head": 1, "tail": 6, "type": "related to procedure"}]}"""
        ),
    },
    {
        "role": "user",
        "content": (
            'Here is a text input: "BACKGROUND: Plantar warts, or verruca plantaris, are common cutaneous lesions of the plantar surface caused by the human papilloma virus (HPV) infection. Current treatments include salicylic acid, cryotherapy, bleomycin, and immunotherapy; however, they have limitations of low-cure rates or side effects. Plantar warts treated with bee venom (BV) have not been reported. Here we present a case of a patient who showed complete remission of multiple plantar warts after BV treatment.\nCASE PRESENTATION: A 13-year-old boy experienced total of 16 plantar warts and visited department of dermatology of Korean Medicine. BV was injected into wart sites. Warts were completely removed at the end of the 8-week treatment with no adverse event recorded. There was improvement during the follow-up after 4 weeks.\nCONCLUSION: This is the first case of plantar warts that improved with BV injection. This study suggests that BV may be a therapeutic option for individuals who cannot receive cryotherapy due to pain during and after treatment or who have refractory or relapsed warts." '
            """Here is the list of input entities: ['BACKGROUND', 'Plantar warts', 'verruca plantaris', 'common cutaneous lesions', 'plantar surface', 'human papilloma virus ( HPV ) infection', 'Current treatments', 'salicylic acid', 'cryotherapy', 'bleomycin', 'immunotherapy', 'limitations', 'low - cure rates', 'side effects', 'Plantar warts', 'bee venom', 'BV', 'case', 'patient', 'complete remission', 'multiple plantar warts', 'BV treatment', '\n CASE PRESENTATION', '13 - year - old boy', 'total of 16 plantar warts', 'department', 'dermatology', 'Korean Medicine', 'BV', 'wart sites', 'Warts', 'end', '8 - week treatment', 'adverse event', 'improvement', 'follow - up', '4 weeks', 'CONCLUSION', 'first case', 'plantar warts', 'BV injection', 'study', 'BV', 'therapeutic option', 'individuals', 'cryotherapy', 'pain', 'treatment', 'refractory or relapsed warts']"""
            "Analyze this text, select and classify the entities, and extract their relationships as per your instructions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            """{"entities": [{"id": 0, "text": "Plantar warts", "type": "Condition"}, {"id": 1, "text": "verruca plantaris", "type": "Synonym"}, {"id": 2, "text": "cutaneous lesions", "type": "Condition"}, {"id": 3, "text": "plantar surface", "type": "Location"}, {"id": 4, "text": "human papilloma virus (HPV)", "type": "Pathogen"}, {"id": 5, "text": "salicylic acid", "type": "Treatment"}, {"id": 6, "text": "cryotherapy", "type": "Treatment"}, {"id": 7, "text": "bleomycin", "type": "Treatment"}, {"id": 8, "text": "immunotherapy", "type": "Treatment"}, {"id": 9, "text": "low-cure rates", "type": "Limitation"}, {"id": 10, "text": "side effects", "type": "Limitation"}, {"id": 11, "text": "bee venom (BV)", "type": "Treatment"}, {"id": 12, "text": "complete remission", "type": "Outcome"}, {"id": 13, "text": "8-week treatment", "type": "Duration"}, {"id": 14, "text": "16 plantar warts", "type": "Condition"}, {"id": 15, "text": "wart sites", "type": "Location"}, {"id": 16, "text": "no adverse event", "type": "Outcome"}, {"id": 17, "text": "follow-up after 4 weeks", "type": "Event"}, {"id": 18, "text": "first case", "type": "Event"}, {"id": 19, "text": "refractory or relapsed warts", "type": "Condition"}, {"id": 20, "text": "pain during and after treatment", "type": "Limitation"}], "relations": [{"head": 0, "tail": 1, "type": "synonym of"}, {"head": 0, "tail": 2, "type": "type of"}, {"head": 0, "tail": 4, "type": "caused by"}, {"head": 0, "tail": 3, "type": "occurs in"}, {"head": 5, "tail": 9, "type": "associated with"}, {"head": 5, "tail": 10, "type": "associated with"}, {"head": 6, "tail": 9, "type": "associated with"}, {"head": 6, "tail": 10, "type": "associated with"}, {"head": 11, "tail": 0, "type": "considered for"}, {"head": 11, "tail": 12, "type": "resulted in"}, {"head": 14, "tail": 11, "type": "treated with"}, {"head": 11, "tail": 15, "type": "administered to"}, {"head": 13, "tail": 16, "type": "resulted in"}, {"head": 18, "tail": 0, "type": "involved"}, {"head": 19, "tail": 11, "type": "treated with"}, {"head": 6, "tail": 20, "type": "associated with"}]}"""
        ),
    },
]
