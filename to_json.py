def convert_to_json(row, data_list):
    data_list.append({
        "id": row['id'],
        "question": row['question'],
        "answer": row['predicted_wiki']
    })
