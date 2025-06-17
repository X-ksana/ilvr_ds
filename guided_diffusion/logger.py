"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
import wandb
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import csv

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

# Global variables for model checkpointing
best_loss = float('inf')
best_ssim = 0.0
last_checkpoint_epoch = 0

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images.
    
    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
    Returns:
        Mean SSIM value across the batch
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Convert from (B, C, H, W) to (B, H, W, C)
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    # Ensure values are in [0, 1]
    img1 = (img1 + 1) / 2.0
    img2 = (img2 + 1) / 2.0
    
    batch_size = img1.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        ssim_val = ssim(
            img1[i], img2[i], 
            data_range=1.0,
            multichannel=True,
            channel_axis=-1
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def save_checkpoint(model, optimizer, epoch, loss, ssim_value, is_best, output_dir):
    """Save model checkpoint.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        ssim_value: Current SSIM value
        is_best: Whether this is the best model so far
        output_dir: Directory to save checkpoints
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'ssim': ssim_value
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best one
    if is_best:
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_model_path)
    
    # Remove old checkpoints if they exist (keep only last 3)
    checkpoints = sorted([f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')])
    if len(checkpoints) > 3:
        for checkpoint_to_remove in checkpoints[:-3]:
            os.remove(os.path.join(output_dir, checkpoint_to_remove))


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    """
    Dumps key/value pairs into CSV files.
    """
    def __init__(self, filename):
        self.filename = filename
        self.keys = []
        self.sep = ','
        self.fp = open(filename, 'w')
        self.writer = None

    def writekvs(self, kvs):
        # Add our row
        extra_keys = list(kvs.keys() - self.keys)
        if extra_keys:
            self.keys.extend(extra_keys)
            self.keys.sort()
        if self.writer is None:
            self.writer = csv.DictWriter(self.fp, fieldnames=self.keys)
            self.writer.writeheader()
        self.writer.writerow(kvs)
        self.fp.flush()

    def close(self):
        self.fp.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


class WandBOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Weights & Biases.
    """
    def __init__(self, project_name, entity=None, config=None):
        """Initialize WandB logger.
        
        Args:
            project_name: Name of the W&B project
            entity: W&B entity (username or team name)
            config: Dictionary of hyperparameters to track
        """
        wandb.init(
            project=project_name,
            entity=entity,
            config=config
        )
        self.last_step = -1  # Track the last step we logged

    def writekvs(self, kvs):
        # Use step from kvs if available, otherwise don't specify step
        step = kvs.get('step', None)
        
        # Skip if this step is less than or equal to the last step we logged
        if step is not None and step <= self.last_step:
            return
        
        # Create a copy without the step for logging
        log_kvs = kvs.copy()
        if 'step' in log_kvs:
            del log_kvs['step']
        
        # Convert all values to float for wandb
        for k, v in sorted(log_kvs.items()):
            if hasattr(v, "dtype"):
                v = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                v = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                v = float(v)
            log_kvs[k] = v
        
        # Ensure SSIM is included in the logs
        if 'ssim' in log_kvs:
            print(f"Logging SSIM value: {log_kvs['ssim']}")  # Debug print
        
        # Log to wandb with step if available
        if step is not None:
            wandb.log(log_kvs, step=int(step))
            self.last_step = int(step)  # Update last step
        else:
            wandb.log(log_kvs)
        
        # Force wandb to sync
        wandb.run.log({}, commit=True)

    def close(self):
        wandb.finish()


def make_output_format(format, ev_dir, log_suffix="", wandb_project=None, wandb_entity=None, wandb_config=None):
    """
    Return a logger for the requested format.
    
    Args:
        format: The requested format to log to ('stdout', 'log', 'json', 'csv', 'tensorboard', 'wandb')
        ev_dir: Directory to store files in
        log_suffix: Suffix to add to files
        wandb_project: Name of the W&B project (only used if format is 'wandb')
        wandb_entity: W&B entity (username or team name)
        wandb_config: Dictionary of hyperparameters to track in W&B
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    elif format == "wandb":
        return WandBOutputFormat(wandb_project, wandb_entity, wandb_config)
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    get_current().set_comm(comm)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm
        self.last_step = -1  # Track the last step we logged

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict

        # Skip if dictionary is empty
        if not d:
            return

        # Get current step
        current_step = d.get('step', None)
        
        # Skip if step is less than or equal to last step
        if current_step is not None and current_step <= self.last_step:
            return
            
        # Update last step
        if current_step is not None:
            self.last_step = current_step

        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))




def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(
    dir=None,
    format_strs=None,
    comm=None,
    log_suffix="",
    wandb_project=None,
    wandb_entity=None,
    wandb_config=None,
):
    """
    Configure the current logger.
    
    Args:
        dir: Directory to save files in
        format_strs: List of output recording formats ('stdout', 'log', 'csv', 'tensorboard', 'wandb')
        comm: MPI communicator
        log_suffix: Suffix to add to files
        wandb_project: Name of the W&B project
        wandb_entity: W&B entity (username or team name)
        wandb_config: Dictionary of hyperparameters to track in W&B
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(dir, exist_ok=True)

    rank = get_rank_without_mpi_import()

    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [
        make_output_format(
            f,
            dir,
            log_suffix,
            wandb_project if f == "wandb" else None,
            wandb_entity if f == "wandb" else None,
            wandb_config if f == "wandb" else None,
        )
        for f in format_strs
    ]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

