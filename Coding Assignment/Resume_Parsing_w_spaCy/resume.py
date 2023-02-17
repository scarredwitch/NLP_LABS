import spacy
from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load('en_core_web_md') #sm &
skill_path = "../../data/skills_educations.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)
nlp.pipe_names

# before that, let's clean our resume.csv dataframe
def preprocessing(sentence):

    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []

    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
                token.pos_ != 'SYM':
            cleaned_tokens.append(token.lemma_.lower().strip())

    return " ".join(cleaned_tokens)

def readPDF(cv_path, page=5):
    
    reader = PdfReader(cv_path)
    page1= reader.pages[page]
    text = page1.extract_text()
    text = preprocessing(text)
    doc = nlp(text)
    print(doc.ents)
    skills = []
    educations = []

    for ent in doc.ents:
        print(ent.label_)
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        if ent.label_ == 'EDUCATION':
            educations.append(ent.text)
    return set(skills), set(educations)

if __name__=="__main__":
    skills, educations = readPDF('../../data/someone_cv.pdf')
    print(educations)