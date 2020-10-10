ENTITY_TYPES = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE', 'SYMPTOM', 'SYNDROME', 'DISEASE_GROUP', 'FOOD',
                'FOOD_GROUP', 'PERSON_GROUP', 'DRUG_GROUP', 'DRUG_DOSAGE', 'DRUG_TASTE', 'DRUG_EFFICACY']

ID2LABEL = ['O',
            '[CLS]',
            '[SEP]',
            '[PAD]',
            'B-DRUG',
            'I-DRUG',
            'B-DRUG_INGREDIENT',
            'I-DRUG_INGREDIENT',
            'B-DISEASE',
            'I-DISEASE',
            'B-SYMPTOM',
            'I-SYMPTOM',
            'B-SYNDROME',
            'I-SYNDROME',
            'B-DISEASE_GROUP',
            'I-DISEASE_GROUP',
            'B-FOOD',
            'I-FOOD',
            'B-FOOD_GROUP',
            'I-FOOD_GROUP',
            'B-PERSON_GROUP',
            'I-PERSON_GROUP',
            'B-DRUG_GROUP',
            'I-DRUG_GROUP',
            'B-DRUG_DOSAGE',
            'I-DRUG_DOSAGE',
            'B-DRUG_TASTE',
            'I-DRUG_TASTE',
            'B-DRUG_EFFICACY',
            'I-DRUG_EFFICACY']


def get_label_dict():
    _tmp = {}
    for i, name in enumerate(ID2LABEL):
        _tmp[name] = i
    return _tmp


LABEL2ID = get_label_dict()

if __name__ == '__main__':
    print(LABEL2ID)