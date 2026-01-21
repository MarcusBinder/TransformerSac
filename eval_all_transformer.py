import os, sys
from evaluate import *
import random



if __name__ == "__main__":
    all_files = os.listdir("/work/users/manils/transformersac/runs")



    runs_integers = list(range(len(all_files)))

    random.shuffle(runs_integers)


    for j in runs_integers:
        file = all_files[j]
        agent_type = file.split("_")[0]
        
        if agent_type == "sac":
            print("Agenttype is sac, we skip this one")
            continue

        elif file.split("_")[1] != 'continue':
            print(f"Skipping {file} as it is not a continue run")
            continue

        elif agent_type == "transformer":
            print(f"Agenttype is transformer, we skip this one")
            continue

        # elif file.split("_")[-1] != 'noreset':
        #     print(f"Skipping {file} as it is not a noreset run")
        #     continue

        elif agent_type == "TransformerSAC":
            print(f"Evaluating {file}")


            checkpoint_dir = os.path.join("/work/users/manils/transformersac/runs", file, "checkpoints")
            layout = 'e'
            episodes = 20
            steps = 250
            deterministic = False
            seed = 42
            output = os.path.join("/work/users/manils/transformersac/evaluation_results", f"{file}_results.nc")
            workers = 25

            # Before running, check if output file already exists
            if os.path.exists(output):
                print(f"Output file {output} already exists. Skipping evaluation.")
                continue

            results = evaluate_checkpoint_dir(
                checkpoint_dir=checkpoint_dir,
                layout=layout,
                num_episodes=episodes,
                num_steps=steps,
                deterministic=deterministic,
                seed=seed,
                output_file=output,
                num_workers=workers,
            )

            results = []

            # evaluate_run(file, "/work/users/manils/transformersac/runs")

        else:
            raise ValueError(f"Unknown agent type {agent_type} in file {file}")
