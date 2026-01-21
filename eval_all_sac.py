import os, sys
from evaluate_mlp import *
import random



if __name__ == "__main__":
    all_files = os.listdir("/work/users/manils/transformersac/runs")



    runs_integers = list(range(len(all_files)))

    random.shuffle(runs_integers)


    for j in runs_integers:
        file = all_files[j]
        agent_type = file.split("_")[0]

        if file.split("_")[2] != "longer":
            print("Only do the longer runs")
            continue
        
        if agent_type == "transformer":
            print("Agenttype is transformer, we skip this one")
            continue

        elif agent_type == "sac":
            print(f"Evaluating {file}")


            # checkpoint_dir = os.path.join("/work/users/manils/transformersac/runs", file, "checkpoints")
            checkpoint_dir = os.path.join("/work/users/manils/transformersac/runs", file)
            # layout = 'e'
            layout = file.split("_")[-2] # get layout from filename
            episodes = 20
            steps = 250
            deterministic = False
            seed = 42
            output = os.path.join("/work/users/manils/transformersac/evaluation_results", f"{file}_results.nc")
            workers = 30

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
