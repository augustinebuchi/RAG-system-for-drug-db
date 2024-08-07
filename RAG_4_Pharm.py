import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import dbX  # Import your custom db module

# Load the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the BlueBERT model for symptom extraction
symptom_extraction_model_name = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
symptom_extraction_tokenizer = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
symptom_extraction_model = pipeline('feature-extraction', model=symptom_extraction_model_name, tokenizer=symptom_extraction_tokenizer)

# Initialize the explanation pipeline
explanation_model_name = 'deepset/roberta-base-squad2'
explanation_pipeline = pipeline('question-answering', model=explanation_model_name, tokenizer=explanation_model_name, max_length=512, truncation=True)

# Connect to the MySQL database using your custom db module
db_manager = dbX.DatabaseManager()
db_manager.connect()  # Establish the connection

# Function to generate explanations
def generate_explanation(context, question):
    response = explanation_pipeline({'context': context, 'question': question})
    return response['answer']

# Function to generate detailed explanations
def generate_detailed_explanation(drug):
    mechanism = drug['mechanism_of_action']
    use = drug['use_details']
    context = f"The drug {drug['name']} is used to {use}. The mechanism of action is {mechanism}."
    question = "How does this drug help with the symptoms and diseases?"
    return generate_explanation(context, question)

# Function to get drug details from the database
def get_drug_details(illness_description_embedding):
    cursor = db_manager.get_database_connection()
    cursor.execute("SELECT * FROM Drugs")
    drugs = cursor.fetchall()

    results = []
    for drug in drugs:
        drug_embedding = embedding_model.encode(drug['description'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(illness_description_embedding, drug_embedding).item()
        drug['similarity'] = similarity
        drug['explanation'] = generate_detailed_explanation(drug)
        results.append(drug)
    
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
    return results

# Function to get medical test details from the database
def get_medical_test_details(illness_description_embedding):
    cursor = db_manager.get_database_connection()
    cursor.execute("SELECT * FROM MedicalTests")
    tests = cursor.fetchall()

    results = []
    for test in tests:
        test_embedding = embedding_model.encode(test['symptoms'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(illness_description_embedding, test_embedding).item()
        test['similarity'] = similarity
        context = f"The test {test['test_name']} is used to diagnose {test['diseases']}. The test measures {test['test_description']}."
        question = "How does this test help with the symptoms and diseases?"
        test['explanation'] = generate_explanation(context, question)
        results.append(test)
    
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
    return results

# Main function
def main():
    illness_description = "Hello doctor, i'm experiencing slight headaches with fever. also experienced weakness yesterday and morning sickness."
    illness_description_embedding = embedding_model.encode(illness_description, convert_to_tensor=True)

    recommended_drugs = get_drug_details(illness_description_embedding)
    recommended_tests = get_medical_test_details(illness_description_embedding)

    print(f"Complaint: {illness_description}")
    print("Top Recommended Drugs:")
    for i, drug in enumerate(recommended_drugs):
        print(f"Option {i + 1}:")
        print(f"Name: {drug['name']}")
        print(f"Type: {drug['type']}")
        print(f"Interacting Drugs: {drug['interacting_drug']}")
        print(f"Interaction Details: {drug['interaction_details']}")
        print(f"Use Details: {drug['use_details']}")
        print(f"Explanation: {drug['explanation']}")
        print()

    print("Top Recommended Medical Tests:")
    for i, test in enumerate(recommended_tests):
        print(f"Option {i + 1}:")
        print(f"Test Name: {test['test_name']}")
        print(f"Test Type: {test['test_type']}")
        print(f"Test Description: {test['test_description']}")
        print(f"Symptoms: {test['symptoms']}")
        print(f"Related Medical Conditions: {test['diseases']}")
        print(f"Explanation: {test['explanation']}")
        print()

if __name__ == "__main__":
    main()
