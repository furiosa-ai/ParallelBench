import time
import wandb

from utils.constants import WANDB_PROJECT_NAME, WANDB_TEAM_NAME


class Logger:
    def __init__(self):
        pass

    def log(self, data, step):
        raise NotImplementedError

    def log_table(self, table_name, df):
        raise NotImplementedError
    
    def finish(self):
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(self, project, entity, name, config, resume=False, exist_ok=True):
        super().__init__()

        cfg_file = config["cfg_file"]
        self.last_ts = None
        self.step = -1

        if not exist_ok:
            api = wandb.Api()
            runs = list(api.runs(project, {"config.cfg_file": cfg_file}))
            if len(runs) > 0:
                raise FileExistsError(f"Found {len(runs)} runs with cfg_file {cfg_file}")

        if resume:
            api = wandb.Api()
            runs = list(api.runs(project, {"config.cfg_file": cfg_file}))
            assert len(runs) == 1, f"Found {len(runs)} runs with cfg_file {cfg_file}"
            wandb.init(project=project, entity=entity, id=runs[0].id, resume="must", config=config, settings=wandb.Settings(init_timeout=1200))
        else:
            wandb.init(project=project, entity=entity, name=name, config=config, settings=wandb.Settings(init_timeout=1200))

    def log_table(self, table_name, df):
        df = df.copy()
        if "output_full" in df.columns:
            df["output_full"] = df["output_full"].astype(str)
        table = wandb.Table(dataframe=df)
        self.log({table_name: table})

    def log(self, data, step=None):
        if step is None:
            self.step += 1
            step = self.step
        else:
            self.step = step

        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()




class ClearMLLogger(Logger):
    def __init__(self, project, entity, name, config, resume=False, exist_ok=True):
        super().__init__()

        cfg_file = config["cfg_file"]
        self.last_ts = None
        self.step = -1

        from clearml import Task
        if not exist_ok:
            assert False
            
            api = Task.get_api()
            runs = api.get_tasks(project_name=project, filters={"config.cfg_file": cfg_file})
            if len(runs) > 0:
                raise FileExistsError(f"Found {len(runs)} runs with cfg_file {cfg_file}")
            
        if resume:
            assert False
            runs = Task.get_tasks(project_name=project, filters={"config.cfg_file": cfg_file})
            assert len(runs) == 1, f"Found {len(runs)} runs with cfg_file {cfg_file}"
            Task.resume(task_id=runs[0].id, project_name=project, task_name=name, config=config)

        else:
            Task.init(project_name=project, task_name=name, config=config, auto_connect_frameworks=False)

    def log_table(self, table_name, df):
        table = wandb.Table(dataframe=df)
        self.log({table_name: table})

    def log(self, data, step=None):
        if step is None:
            self.step += 1
            step = self.step
        else:
            self.step = step

        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()


class NoLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def log_table(self, table_name, df):
        pass

    def log(self, data, step=None):
        pass

    def finish(self):
        pass


def create_logger(logger_type, cfg, resume=False, run_prefix=None, **kwargs):
    # run_name = cfg["cfg_file"].split("/", 2)[-1].replace("/", "_").replace(".yaml", "")
    run_name = cfg["cfg_file"].replace("cfg/", "").replace("temp/", "").replace(".yaml", "")

    if run_prefix is not None:
        run_name = f"{run_prefix}/{run_name}"

    logger_class = {
        "wandb": WandbLogger,
        "clearml": None,
        "none": NoLogger,
    }[logger_type]

    logger = logger_class(project=WANDB_PROJECT_NAME, entity=WANDB_TEAM_NAME, name=run_name, config=cfg, resume=resume, **kwargs)
    return logger