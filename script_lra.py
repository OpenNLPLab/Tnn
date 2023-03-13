import os
import sys
import time

# change
gtu_use_decay = True
PREFIX = "your path to lra"

batches = {
    "cifar": 200,
    "imdb": 32,
    "listops": 128,
    "pathfinder": 128,
    "pathfinderx": 64,
    "aan": 64,
}

gpus = {
    "cifar": 4,
    "imdb": 2,
    "listops": 2,
    "pathfinder": 4,
    "pathfinderx": 8,
    "aan": 2,
}

d_model_dict = {
    "cifar": 128,
    "imdb": 128,
    "listops": 128,
    "pathfinder": 64,
    "pathfinderx": 64,
    "aan": 64,
}

n_layers_dict = {
    "cifar": 12,
    "imdb": 4,
    "listops": 4,
    "pathfinder": 6,
    "pathfinderx": 6,
    "aan": 2,
}

expand_ratio_gtu_dict = {
    "cifar": 2,
    "imdb": 1,
    "listops": 1,
    "pathfinder": 3,
    "pathfinderx": 3,
    "aan": 1,
}

expand_ratio_glu_dict = {
    "cifar": 3,
    "imdb": 2,
    "listops": 1,
    "pathfinder": 2.5,
    "pathfinderx": 2.5,
    "aan": 1.5,
}

gtu_gamma = {
    "cifar": 0.7,
    "imdb": 0.9,
    "listops": 1,
    "pathfinder": [1],
    "pathfinderx": [0.999],
    "aan": [0.6],
}

norm_dict = {
    "cifar": "synbatch",
    "imdb": "synbatch",
    "listops": "synbatch",
    "pathfinder": "synbatch",
    "pathfinderx": "synbatch",
    "aan": "synbatch",
}

lr_dict = {
    "cifar": 0.007,
    "imdb": [0.00105],
    "listops": 0.0005,
    "pathfinder": [0.001],
    "pathfinderx": [0.0001],
    "aan": [0.01],
}

wd_dict = {
    "cifar": [0.001],
    "imdb": 0.001,
    "listops": 0.1,
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": [0.01],
}

dropout_dict = {
    "cifar": [0.1],
    "imdb": 0.1,
    "listops": 0,
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": 0,
}

dpb_type_dict = {
    "cifar": 1,
    "imdb": 1,
    "listops": 1,
    "pathfinder": 1,
    "pathfinderx": 1,
    "aan": 1,
}

dpb_layers_dict = {
    "cifar": 0,
    "imdb": 3,
    "listops": 3,
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": 3,
}

gtu_dpb_dim_dict = {
    "cifar": 16,
    "imdb": 32,
    "listops": 32,
    "pathfinder": 16,
    "pathfinderx": 16,
    "aan": 16,
}

prenorm_dict = {
    "cifar": True,
    "imdb": True,
    "listops": True,
    "pathfinder": True,
    "pathfinderx": True,
    "aan": True,
}

warmup_steps_dict = {
    "cifar": [30000],
    "imdb": 3000,
    "listops": [1000],
    "pathfinder": [5000],
    "pathfinderx": 312,
    "aan": [25000],
}


# for these tasks, you should use tno
tasks = ["aan"]
tasks = ["imdb"]
tasks = ["listops"]
archs = ["tno"]

# for these tasks, you should use tno2d
tasks = ["cifar"]
tasks = ["pathfinder"]
tasks = ["pathfinderx"]
archs = ["tno2d"]


def to_iter(*args):
    n = len(args)
    new_args = []
    for i in range(n):
        if not isinstance(args[i], list):
            arg = [args[i]]
        else:
            arg = args[i]
        new_args.append(arg)

    return helper(*new_args)


def helper(*args):
    n = len(args)
    if n == 1:
        res = [[arg] for arg in args[0]]
        return res
    else:
        arr = helper(*args[1:])
        res = []
        for par in args[0]:
            for data in arr:
                res.append([par] + list(data))
        return res


for i, task in enumerate(tasks):
    pars = to_iter(
        archs,
        n_layers_dict[task],
        expand_ratio_gtu_dict[task],
        expand_ratio_glu_dict[task],
        d_model_dict[task],
        gtu_gamma[task],
        batches[task],
        norm_dict[task],
        lr_dict[task],
        wd_dict[task],
        dropout_dict[task],
        dpb_type_dict[task],
        dpb_layers_dict[task],
        gtu_dpb_dim_dict[task],
        prenorm_dict[task],
        warmup_steps_dict[task],
    )
    print(pars)
    print(task)
    print(len(pars))
    time.sleep(10)
    for (
        arch,
        n_layers,
        expand_ratio_gtu,
        expand_ratio_glu,
        d_model,
        gamma,
        total_batch,
        norm,
        lr,
        wd,
        dropout,
        dpb_type,
        dpb_layers,
        gtu_dpb_dim,
        prenorm,
        warmup_steps,
    ) in pars:
        if task == "imdb":
            seq_len = 4096
            if not gtu_use_decay:
                os.system(
                    f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                )
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                for i in range(1):
                    print("imdb lr: ", lr)
                    time.sleep(10)
                    pid = os.fork()
                    if pid == 0:
                        os.system(
                            f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                        )
                        sys.exit(0)
        elif task == "cifar":
            seq_len = 1024
            if not gtu_use_decay:
                os.system(
                    f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}"
                )
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("cifar lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
        elif task == "listops":
            seq_len = 2048
            if not gtu_use_decay:
                os.system(
                    f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}"
                )
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("listops lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
        elif task == "pathfinder":
            seq_len = 1024
            if not gtu_use_decay:
                print("pathfinder lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("pathfinder lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
        elif task == "pathfinderx":
            seq_len = 128 * 128
            if not gtu_use_decay:
                print("pathfinderx lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu}"
                    )
                    sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("pathfinderx lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
        elif task == "aan":
            seq_len = 4000
            if not gtu_use_decay:
                os.system(
                    f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} 0 {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                )
                sys.exit(0)
            else:
                gpu = gpus[task]
                batch = total_batch // gpu
                workers = gpu * 20
                print("aan lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/run_task.sh {task} {arch} {batch} {n_layers} {d_model} {gtu_dpb_dim} {norm} {expand_ratio_gtu} {expand_ratio_glu} {gtu_use_decay} {gamma} {lr} {wd} {gpu} {workers} {dropout} {dpb_type} {dpb_layers} {prenorm} {warmup_steps}"
                    )
                    sys.exit(0)
