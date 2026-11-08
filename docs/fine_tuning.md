# Fine-tuning Dream-1 on your demonstrations

Fine-tuning trains a copy of Dream-1 on your dataset through the Dream API. The result is a model of your own, with an id like `dream-1-ft:a1b2c3d4-myteam-cubes`, that you then run on the arms. `tools/finetune.py` runs every step of Dream's [fine-tuning tutorial](https://platform.dreamlabs.ai/api-documentation/fine-tuning/) in one command; the tutorial explains each step in depth.

## Before you start

- **The Dream library** is already installed on your station, in `~/dream-env`. The tools below run there, not in this repo's environment. (Outside the hackathon, the [Getting started page](https://platform.dreamlabs.ai/api-documentation/) installs it.)
- **Your team's API key**, saved on the station. Create one on the platform's [API Keys](https://platform.dreamlabs.ai/api-keys) page, then run step 2 of the [Getting started page](https://platform.dreamlabs.ai/api-documentation/) on the station: it asks for the key without showing it, checks it with the platform and saves it to `~/.dreamlabs/credentials`. Your datasets, jobs and models then belong to your team's account.
- **Credits**: a job's price grows with its training steps, so it is not a fixed amount. At the time of writing it is $100 for 500 steps, $200 for 1,000 (the default, and what `tools/finetune.py` runs) and $400 for 2,000 ([What it costs](https://platform.dreamlabs.ai/api-documentation/fine-tuning/#what-it-costs)); the platform's Fine-tune form shows the current price, and your balance is on the [Billing](https://platform.dreamlabs.ai/billing) page. Check with the organizers before you queue one.
- **Time**: about 7 to 8 hours of training, plus any wait in the queue, and up to an hour for your model's first start.

## 1. Metal and Maker arms: add the tool poses

Dream's models for MakerMods arms train on each gripper's pose beside the joints. `tools/add_eef.py` computes those poses with Dream's own model of your arm, the same one `dreamcli` uses when it runs your model. It writes a new dataset next to yours, `<name>-eef`, and checks it:

```bash
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/myteam-cubes --arm metal    # or --arm maker
```

If you collect more episodes afterwards, run it again with `--replace` to rebuild the `-eef` folder. YAM datasets need no extra step: their fine-tunes train on the joints.

## 2. Upload and fine-tune

```bash
# YAM
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes --arm yam --name myteam-cubes --suffix myteam-cubes
# Metal or Maker arm: the -eef dataset
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/myteam-cubes-eef --arm metal --name myteam-cubes --suffix myteam-cubes
```

- `--name` is the dataset's name on the platform, and `--suffix` the end of your model's id. Both use lowercase letters, digits and hyphens, and each must be new on your account.
- The first run on an account registers the robot's embodiment, the description of what each recorded value means, as `hackathon-yam`, `hackathon-metal` or `hackathon-maker`, and later runs reuse it.
- `--lr-schedule cosine` tends to suit a model trained on a single task; `--learning-rate` takes 1e-6 to 1e-4 (the default is 1e-5).

The command checks the dataset on your computer first, then uploads it, waits for the platform to process it, queues the job and waits for your model, printing the status as it goes. It ends with your model's id and the command to try it. The wait lasts hours, so run it inside tmux on the station (`tmux new -A -s finetune`). The job trains on the platform whatever happens to your terminal.

You can stop it with Ctrl+C at any point once the upload is done. Pick up again with the id it printed:

```bash
~/dream-env/bin/python tools/finetune.py --arm yam --dataset-id ds_... --suffix myteam-cubes   # after the upload
~/dream-env/bin/python tools/finetune.py --arm yam --job-id ftj_...                       # after the job is queued
```

A queued job runs to its end and cannot be cancelled. If training fails, the job ends with no model and costs nothing; the error says why.

### Upload without fine-tuning

To put a dataset on your account without queueing a job, run `tools/upload_dataset.py`. It runs the same checks and the same upload as `tools/finetune.py` (the dataset folder is the same one, the `-eef` folder on Metal and Maker arms), waits until the platform has processed the dataset, and stops. Nothing is charged:

```bash
~/dream-env/bin/python tools/upload_dataset.py ~/hackathon_data/myteam-cubes --arm yam --name myteam-cubes
```

It prints the dataset's id. Fine-tune on it later with `tools/finetune.py --arm yam --dataset-id ds_... --suffix myteam-cubes` (that queues the paid job). A dataset name must be new on your account, so a later upload of more episodes needs another `--name`.

## Several sessions, several tasks

Every session you collect into the same folder becomes one dataset, and one fine-tune can learn several tasks. Labels are optional: Dream's service labels every dataset from its overhead camera when it processes it, and a `--task` text you recorded travels with the episodes too. To train on two folders together, put both inside one parent folder and upload the parent: the platform reads each subfolder as one session. On Metal and Maker arms, run `tools/add_eef.py` on each folder first and put the `-eef` folders in the parent.

Next: [run your model](running_your_model.md).
