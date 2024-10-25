from openai import OpenAI

def query_llm(instruction):
    try:
        client = OpenAI(
            base_url="http://192.168.50.155:8000/v1/",
            api_key="token-abc123",
        )

        completion = client.chat.completions.create(
            model="google/gemma-2-2b-it",
            seed=1337,
            temperature=0.0,
            messages=[
                {"role": "user", "content": instruction}
            ]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    


def query_constrained_llm(instruction, values):
    try:
        client = OpenAI(
            base_url="http://192.168.50.155:8000/v1/",
            api_key="token-abc123",
        )

        completion = client.chat.completions.create(
        model="google/gemma-2-2b-it",
        messages=[
            {"role": "user", "content": instruction}
        ],
        extra_body={
            "guided_choice": values
            }
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"


def batch_process_transcripts(df, instruction, column_name, output_column, values=None):
    try:
        client = OpenAI(
            base_url="http://192.168.50.155:8000/v1/",
            api_key="token-abc123",
        )
        
        # Convert comma-separated string to list and strip whitespace
        if values:
            values = [v.strip() for v in values.split(',')]
        
        results = []
        for idx, row in df.iterrows():
            transcript = str(row[column_name])[:500]
            prompt = f"{instruction}\n\nTranscript: {transcript}"
            
            if values:
                completion = client.chat.completions.create(
                    model="google/gemma-2-2b-it",
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"guided_choice": values}
                )
            else:
                completion = client.chat.completions.create(
                    model="google/gemma-2-2b-it",
                    messages=[{"role": "user", "content": prompt}]
                )
                
            results.append(completion.choices[0].message.content)
            
        df[output_column] = results
        return df, "Processing completed successfully"
        
    except Exception as e:
        return None, f"Error in batch processing: {str(e)}"