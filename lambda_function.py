import json
import boto3
import random

# Importer vos algorithmes d'ordonnancement locaux (ils seront inclus dans le .zip)
from peft_scheduler import PEFT
from heft_scheduler import HEFT

s3_client = boto3.client('s3')

def adapt_graph_data(data, num_processors=4, seed=42):
    """
    Adapte le JSON généré par graph_generator pour les algorithmes PEFT / HEFT.
    """
    random.seed(seed)
    processors = [f"core_{i}" for i in range(num_processors)]
    tasks = {}
    
    for task_info in data["tasks"]:
        raw_id = task_info["id"]
        # Convert "task1" to "T1" pour le tri dans PEFT/HEFT
        if raw_id.startswith("task"):
            task_id = "T" + raw_id[4:]
        else:
            task_id = str(raw_id)
            
        t_duration = float(task_info["duration"])
        
        comp_costs = {}
        for p in processors:
            variation = random.uniform(0.7, 1.3)
            comp_costs[p] = round(t_duration * variation, 2)
            
        dependencies = {}
        for dep_id in task_info.get("dependencies", []):
             if str(dep_id).startswith("task"):
                 norm_dep = "T" + str(dep_id)[4:]
             else:
                 norm_dep = str(dep_id)
             dependencies[norm_dep] = round(random.uniform(5.0, 20.0), 2)
             
        tasks[task_id] = {
            "comp_costs": comp_costs,
            "dependencies": dependencies
        }
        
    return {"processors": processors, "tasks": tasks}

def lambda_handler(event, context):
    """
    Fonction AWS Lambda Principale.
    """
    # Noms de votre bucket et vos chemins (d'après la capture d'écran)
    bucket_name = 'central-supelec-data-group2'
    input_key = 'input_data/graph.json'
    output_key = 'output_data/ordonnancement.json'
    
    try:
        print(f"Fetching {input_key} from bucket {bucket_name}...")
        
        # 1. Lire les données d'entrée depuis le S3
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        input_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # 2. Adapter les données (générer les coûts hétérogènes et de communication)
        dag_data = adapt_graph_data(input_data, num_processors=4, seed=42)
        
        # 3. Exécuter l'algorithme d'ordonnancement (PEFT ici)
        print("Running PEFT algorithm...")
        scheduler = PEFT(dag_data['tasks'], dag_data['processors'])
        tasks_schedule, final_makespan = scheduler.schedule()
        
        # 4. Formater les résultats pour correspondre à ordonnancement_local.json
        output_format = {p: [] for p in dag_data['processors']}
        
        for res in tasks_schedule:
            # Reconvertir 'T1' en 'task1' pour la sortie
            t_id = res['Task ID']
            original_task_id = "task" + t_id[1:] if t_id.startswith("T") else t_id
            
            output_format[res['Assigned Processor']].append({
                "task": original_task_id,
                "start_time": res['Start Time'],
                "end_time": res['End Time']
            })
            
        # 5. Sauvegarder les résultats de sortie dans le bucket S3
        print(f"Uploading results to {output_key}...")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=json.dumps(output_format, indent=2),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Ordonnancement terminé avec succès, makespan de {final_makespan:.2f}')
        }
        
    except Exception as e:
        print(f"Erreur déclenchée: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Erreur lors du traitement: {str(e)}')
        }
