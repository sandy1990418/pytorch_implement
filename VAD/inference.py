import logging
import hashlib
import random
import re
import sys
import warnings
import torch
import torchaudio
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.nn import DataParallel as DP
from copy import copy
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import chain

from VAD.Speechbrain_VAD.hyperpyyaml import load_hyperpyyaml
from VAD.Speechbrain_VAD.utils.data_utils import split_path
from VAD.Speechbrain_VAD.dataio.preprocess import AudioNormalizer
from VAD.Speechbrain_VAD.utils.data_utils import split_path
from VAD.Speechbrain_VAD.utils.distributed import run_on_main
from VAD.Speechbrain_VAD.pretrained.fetching import fetch

logger = logging.getLogger(__name__)



class Pretrained(torch.nn.Module):
    """Takes a trained model and makes predictions on new data.

    This is a base class which handles some common boilerplate.
    It intentionally has an interface similar to ``Brain`` - these base
    classes handle similar things.

    Subclasses of Pretrained should implement the actual logic of how
    the pretrained system runs, and add methods with descriptive names
    (e.g. transcribe_file() for ASR).

    Pretrained is a torch.nn.Module so that methods like .to() or .eval() can
    work. Subclasses should provide a suitable forward() implementation: by
    convention, it should be a method that takes a batch of audio signals and
    runs the full model (as applicable).

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        The Torch modules that make up the learned system. These can be treated
        in special ways (put on the right device, frozen, etc.). These are available
        as attributes under ``self.mods``, like self.mods.model(x)
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        Options parsed from command line. See ``speechbrain.parse_arguments()``.
        List that are supported here:
         * device
         * data_parallel_count
         * data_parallel_backend
         * distributed_launch
         * distributed_backend
         * jit
         * jit_module_keys
         * compule
         * compile_module_keys
         * compile_mode
         * compile_using_fullgraph
         * compile_using_dynamic_shape_tracing
    freeze_params : bool
        To freeze (requires_grad=False) parameters or not. Normally in inference
        you want to freeze the params. Also calls .eval() on all modules.
    """

    HPARAMS_NEEDED = []
    MODULES_NEEDED = []

    def __init__(
        self, modules=None, hparams=None, run_opts=None, freeze_params=True
    ):
        super().__init__()
        # Arguments passed via the run opts dictionary. Set a limited
        # number of these, since some don't apply to inference.
        run_opt_defaults = {
            "device": "cpu",
            "data_parallel_count": -1,
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit": False,
            "jit_module_keys": None,
            "compile": False,
            "compile_module_keys": None,
            "compile_mode": "reduce-overhead",
            "compile_using_fullgraph": False,
            "compile_using_dynamic_shape_tracing": False,
        }
        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Put modules on the right device, accessible with dot notation
        self.mods = torch.nn.ModuleDict(modules)
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        # Check MODULES_NEEDED and HPARAMS_NEEDED and
        # make hyperparams available with dot notation
        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            # Also first check that all required params are found:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        # Prepare modules for computation, e.g. jit
        self._prepare_modules(freeze_params)

        # Audio normalization
        self.audio_normalizer = hparams.get(
            "audio_normalizer", AudioNormalizer()
        )

    def _prepare_modules(self, freeze_params):
        """Prepare modules for computation, e.g. jit.

        Arguments
        ---------
        freeze_params : bool
            Whether to freeze the parameters and call ``eval()``.
        """

        # Make jit-able
        self._compile()
        self._wrap_distributed()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False

    def load_audio(self, path, savedir="audio_cache", **kwargs):
        """Load an audio file with this model's input spec

        When using a speech model, it is important to use the same type of data,
        as was used to train the model. This means for example using the same
        sampling rate and number of channels. It is, however, possible to
        convert a file from a higher sampling rate to a lower one (downsampling).
        Similarly, it is simple to downmix a stereo file to mono.
        The path can be a local path, a web url, or a link to a huggingface repo.
        """
        source, fl = split_path(path)
        kwargs = copy(kwargs)  # shallow copy of references only
        channels_first = kwargs.pop(
            "channels_first", False
        )  # False as default value: SB consistent tensor format
        if kwargs:
            fetch_kwargs = dict()
            for key in [
                "overwrite",
                "save_filename",
                "use_auth_token",
                "revision",
                "cache_dir",
                "silent_local_fetch",
            ]:
                if key in kwargs:
                    fetch_kwargs[key] = kwargs.pop(key)
            path = fetch(fl, source=source, savedir=savedir, **fetch_kwargs)
        else:
            path = fetch(fl, source=source, savedir=savedir)
        signal, sr = torchaudio.load(
            str(path), channels_first=channels_first, **kwargs
        )
        return self.audio_normalizer(signal, sr)

    def _compile(self):
        """Compile requested modules with either JIT or TorchInductor."""
        compile_available = hasattr(torch, "compile")

        if not compile_available and self.compile_module_keys is not None:
            raise ValueError(
                "'compile_module_keys' specified, but this install of PyTorch "
                "seems to be too old to support it."
            )

        # Modules to compile with torch.compile
        compile_module_keys = set()
        if self.compile:
            if self.compile_module_keys is None:
                compile_module_keys = set(self.mods)
            else:
                compile_module_keys = set(self.compile_module_keys)
                logger.warning(
                    "--compile and --compile_module_keys are both specified. "
                    "Only modules specified in --compile_module_keys will be compiled."
                )

        # Modules to compile with jit
        jit_module_keys = set()
        if self.jit:
            if self.jit_module_keys is None:
                jit_module_keys = set(self.mods)
            else:
                jit_module_keys = set(self.jit_module_keys)
                logger.warning(
                    "--jit and --jit_module_keys are both specified. "
                    "Only modules specified in --jit_module_keys will be compiled."
                )

        # find missing keys
        for name in compile_module_keys | jit_module_keys:
            if name not in self.mods:
                raise ValueError(
                    f"module {name} is not defined in your hparams file."
                )

        # try 'torch.compile', remove successful compiles from JIT list
        for name in compile_module_keys:
            try:
                module = torch.compile(
                    self.mods[name],
                    mode=self.compile_mode,
                    fullgraph=self.compile_using_fullgraph,
                    dynamic=self.compile_using_dynamic_shape_tracing,
                )
            except Exception as e:
                logger.warning(
                    f"'{name}' in 'compile_module_keys' failed to compile "
                    f"and will be skipped (may fallback onto JIT, if "
                    f"specified): {e}"
                )
                continue

            self.mods[name] = module.to(self.device)
            jit_module_keys.discard(name)

        for name in jit_module_keys:
            module = torch.jit.script(self.mods[name])
            self.mods[name] = module.to(self.device)

    def _compile_jit(self):
        warnings.warn("'_compile_jit' is deprecated; use '_compile' instead")
        self._compile()

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # for ddp, all module must run on same GPU
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(module, device_ids=[self.device])
                    self.mods[name] = module
        else:
            # data_parallel_backend
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # if distributed_count = -1 then use all gpus
                    # otherwise, specify the set of gpu to use
                    if self.data_parallel_count == -1:
                        module = DP(module)
                    else:
                        module = DP(
                            module, [i for i in range(self.data_parallel_count)]
                        )
                    self.mods[name] = module

    @classmethod
    def from_hparams(
        cls,
        source,
        hparams_file="hyperparams.yaml",
        pymodule_file="custom.py",
        overrides={},
        savedir=None,
        use_auth_token=False,
        revision=None,
        download_only=False,
        **kwargs,
    ):
        """Fetch and load based from outside source based on HyperPyYAML file

        The source can be a location on the filesystem or online/huggingface

        You can use the pymodule_file to include any custom implementations
        that are needed: if that file exists, then its location is added to
        sys.path before Hyperparams YAML is loaded, so it can be referenced
        in the YAML.

        The hyperparams file should contain a "modules" key, which is a
        dictionary of torch modules used for computation.

        The hyperparams file should contain a "pretrainer" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        Arguments
        ---------
        source : str or Path or FetchSource
            The location to use for finding the model. See
            ``speechbrain.pretrained.fetching.fetch`` for details.
        hparams_file : str
            The name of the hyperparameters file to use for constructing
            the modules necessary for inference. Must contain two keys:
            "modules" and "pretrainer", as described.
        pymodule_file : str
            A Python file can be fetched. This allows any custom
            implementations to be included. The file's location is added to
            sys.path before the hyperparams YAML file is loaded, so it can be
            referenced in YAML.
            This is optional, but has a default: "custom.py". If the default
            file is not found, this is simply ignored, but if you give a
            different filename, then this will raise in case the file is not
            found.
        overrides : dict
            Any changes to make to the hparams file when it is loaded.
        savedir : str or Path
            Where to put the pretraining material. If not given, will use
            ./pretrained_models/<class-name>-hash(source).
        use_auth_token : bool (default: False)
            If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
            default is False because the majority of models are public.
        revision : str
            The model revision corresponding to the HuggingFace Hub model revision.
            This is particularly useful if you wish to pin your code to a particular
            version of a model hosted at HuggingFace.
        download_only : bool (default: False)
            If true, class and instance creation is skipped.
        """
        if savedir is None:
            clsname = cls.__name__
            savedir = f"./pretrained_models/{clsname}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
        hparams_local_path = fetch(
            filename=hparams_file,
            source=source,
            savedir=savedir,
            overwrite=False,
            save_filename=None,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        try:
            pymodule_local_path = fetch(
                filename=pymodule_file,
                source=source,
                savedir=savedir,
                overwrite=False,
                save_filename=None,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir='VAD'
            )
            sys.path.append(str(pymodule_local_path.parent))
        except ValueError:
            if pymodule_file == "custom.py":
                # The optional custom Python module file did not exist
                # and had the default name
                pass
            else:
                # Custom Python module file not found, but some other
                # filename than the default was given.
                raise

        # Load the modules:
        with open(hparams_local_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Pretraining:
        pretrainer = hparams["pretrainer"]
        pretrainer.set_collect_in(savedir)
        # For distributed setups, have this here:
        run_on_main(
            pretrainer.collect_files, kwargs={"default_source": source},
        )
        # Load on the CPU. Later the params can be moved elsewhere by specifying
        if not download_only:
            # run_opts={"device": ...}
            pretrainer.load_collected(device="cpu")

            # Now return the system
            return cls(hparams["modules"], hparams, **kwargs)


class VAD(Pretrained):
    """A ready-to-use class for Voice Activity Detection (VAD) using a
    pre-trained model.

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import VAD
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> VAD = VAD.from_hparams(
    ...     source="speechbrain/vad-crdnn-libriparty",
    ...     savedir=tmpdir,
    ... )

    >>> # Perform VAD
    >>> boundaries = VAD.get_speech_segments("tests/samples/single-mic/example1.wav")
    """

    HPARAMS_NEEDED = ["sample_rate", "time_resolution", "device"]

    MODULES_NEEDED = ["compute_features", "mean_var_norm", "model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_resolution = self.hparams.time_resolution
        self.sample_rate = self.hparams.sample_rate

    def get_speech_prob_file(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
    ):
        """Outputs the frame-level speech probability of the input audio file
        using the neural model specified in the hparam file. To make this code
        both parallelizable and scalable to long sequences, it uses a
        double-windowing approach.  First, we sequentially read non-overlapping
        large chunks of the input signal.  We then split the large chunks into
        smaller chunks and we process them in parallel.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size:
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            True, creates overlapped small chunks. The probabilities of the
            overlapped chunks are combined using hamming windows.

        Returns
        -------
        prob_vad: torch.Tensor
            Tensor containing the frame-level speech probabilities for the
            input audio file.
        """
        # Getting the total size of the input file
        sample_rate, audio_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the length (in samples) of the large and small chunks
        long_chunk_len = int(sample_rate * large_chunk_size)
        small_chunk_len = int(sample_rate * small_chunk_size)

        # Setting the step size of the small chunk (50% overlapping windows are supported)
        small_chunk_step = small_chunk_size
        if overlap_small_chunk:
            small_chunk_step = small_chunk_size / 2

        # Computing the length (in sample) of the small_chunk step size
        small_chunk_len_step = int(sample_rate * small_chunk_step)

        # Loop over big chunks
        prob_chunks = []
        last_chunk = False
        begin_sample = 0
        while True:
            # Reading the big chunk
            large_chunk, fs = torchaudio.load(
                str(audio_file),
                frame_offset=begin_sample,
                num_frames=long_chunk_len,
            )
            large_chunk = large_chunk.to(self.device)

            # Manage padding of the last small chunk
            if last_chunk or large_chunk.shape[-1] < small_chunk_len:
                padding = torch.zeros(
                    1, small_chunk_len, device=large_chunk.device
                )
                large_chunk = torch.cat([large_chunk, padding], dim=1)

            # Splitting the big chunk into smaller (overlapped) ones
            small_chunks = torch.nn.functional.unfold(
                large_chunk.unsqueeze(1).unsqueeze(2),
                kernel_size=(1, small_chunk_len),
                stride=(1, small_chunk_len_step),
            )
            small_chunks = small_chunks.squeeze(0).transpose(0, 1)

            # Getting (in parallel) the frame-level speech probabilities
            small_chunks_prob = self.get_speech_prob_chunk(small_chunks)
            small_chunks_prob = small_chunks_prob[:, :-1, :]

            # Manage overlapping chunks
            if overlap_small_chunk:
                small_chunks_prob = self._manage_overlapped_chunks(
                    small_chunks_prob
                )

            # Prepare for folding
            small_chunks_prob = small_chunks_prob.permute(2, 1, 0)

            # Computing lengths in samples
            out_len = int(
                large_chunk.shape[-1] / (sample_rate * self.time_resolution)
            )
            kernel_len = int(small_chunk_size / self.time_resolution)
            step_len = int(small_chunk_step / self.time_resolution)

            # Folding the frame-level predictions
            small_chunks_prob = torch.nn.functional.fold(
                small_chunks_prob,
                output_size=(1, out_len),
                kernel_size=(1, kernel_len),
                stride=(1, step_len),
            )

            # Appending the frame-level speech probabilities of the large chunk
            small_chunks_prob = small_chunks_prob.squeeze(1).transpose(-1, -2)
            prob_chunks.append(small_chunks_prob)

            # Check stop condition
            if last_chunk:
                break

            # Update counter to process the next big chunk
            begin_sample = begin_sample + long_chunk_len

            # Check if the current chunk is the last one
            if begin_sample + long_chunk_len > audio_len:
                last_chunk = True

        # Converting the list to a tensor
        prob_vad = torch.cat(prob_chunks, dim=1)
        last_elem = int(audio_len / (self.time_resolution * sample_rate))
        prob_vad = prob_vad[:, 0:last_elem, :]

        return prob_vad

    def _manage_overlapped_chunks(self, small_chunks_prob):
        """This support function manages overlapped the case in which the
        small chunks have a 50% overlap."""

        # Weighting the frame-level probabilities with a hamming window
        # reduces uncertainty when overlapping chunks are used.
        hamming_window = torch.hamming_window(
            small_chunks_prob.shape[1], device=self.device
        )

        # First and last chunks require special care
        half_point = int(small_chunks_prob.shape[1] / 2)
        small_chunks_prob[0, half_point:] = small_chunks_prob[
            0, half_point:
        ] * hamming_window[half_point:].unsqueeze(1)
        small_chunks_prob[-1, 0:half_point] = small_chunks_prob[
            -1, 0:half_point
        ] * hamming_window[0:half_point].unsqueeze(1)

        # Applying the window to all the other probabilities
        small_chunks_prob[1:-1] = small_chunks_prob[
            1:-1
        ] * hamming_window.unsqueeze(0).unsqueeze(2)

        return small_chunks_prob

    def get_speech_prob_chunk(self, wavs, wav_lens=None):
        """Outputs the frame-level posterior probability for the input audio chunks
        Outputs close to zero refers to time steps with a low probability of speech
        activity, while outputs closer to one likely contain speech.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        outputs = self.mods.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs, h = self.mods.rnn(outputs)
        outputs = self.mods.dnn(outputs)
        output_prob = torch.sigmoid(outputs)

        return output_prob

    def apply_threshold(
        self, vad_prob, activation_th=0.5, deactivation_th=0.25
    ):
        """Scans the frame-level speech probabilities and applies a threshold
        on them. Speech starts when a value larger than activation_th is
        detected, while it ends when observing a value lower than
        the deactivation_th.

        Arguments
        ---------
        vad_prob: torch.Tensor
            Frame-level speech probabilities.
        activation_th:  float
            Threshold for starting a speech segment.
        deactivation_th: float
            Threshold for ending a speech segment.

        Returns
        -------
        vad_th: torch.Tensor
            Tensor containing 1 for speech regions and 0 for non-speech regions.
        """
        vad_activation = (vad_prob >= activation_th).int()
        vad_deactivation = (vad_prob >= deactivation_th).int()
        vad_th = vad_activation + vad_deactivation

        # Loop over batches and time steps
        for batch in range(vad_th.shape[0]):
            for time_step in range(vad_th.shape[1] - 1):
                if (
                    vad_th[batch, time_step] == 2
                    and vad_th[batch, time_step + 1] == 1
                ):
                    vad_th[batch, time_step + 1] = 2

        vad_th[vad_th == 1] = 0
        vad_th[vad_th == 2] = 1
        return vad_th

    def get_boundaries(self, prob_th, output_value="seconds"):
        """Computes the time boundaries where speech activity is detected.
        It takes in input frame-level binary decisions
        (1 for speech, 0 for non-speech) and outputs the begin/end second
        (or sample) of each detected speech region.

        Arguments
        ---------
        prob_th: torch.Tensor
            Frame-level binary decisions (1 for speech frame, 0 for a
            non-speech one).  The tensor can be obtained from apply_threshold.
        output_value: 'seconds' or 'samples'
            When the option 'seconds' is set, the returned boundaries are in
            seconds, otherwise, it reports them in samples.

        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second (or sample) of speech segments
            in even positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """
        # Shifting frame-levels binary decision by 1
        # This allows detecting changes in speech/non-speech activities
        prob_th_shifted = torch.roll(prob_th, dims=1, shifts=1)
        prob_th_shifted[:, 0, :] = 0
        prob_th = prob_th + prob_th_shifted

        # Needed to first and last time step
        prob_th[:, 0, :] = (prob_th[:, 0, :] >= 1).int()
        prob_th[:, -1, :] = (prob_th[:, -1, :] >= 1).int()

        # Fix edge cases (when a speech starts in the last frames)
        if (prob_th == 1).nonzero().shape[0] % 2 == 1:
            prob_th = torch.cat(
                (
                    prob_th,
                    torch.Tensor([1.0])
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .to(self.device),
                ),
                dim=1,
            )

        # Where prob_th is 1 there is a change
        indexes = (prob_th == 1).nonzero()[:, 1].reshape(-1, 2)

        # Remove 1 from end samples
        indexes[:, -1] = indexes[:, -1] - 1

        # From indexes to samples
        seconds = (indexes * self.time_resolution).float()
        samples = (self.sample_rate * seconds).round().int()

        if output_value == "seconds":
            boundaries = seconds
        else:
            boundaries = samples
        return boundaries

    def merge_close_segments(self, boundaries, close_th=0.250):
        """Merges segments that are shorter than the given threshold.

        Arguments
        ---------
        boundaries : str
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.

        Returns
        -------
        new_boundaries
            The new boundaries with the merged segments.
        """

        new_boundaries = []

        # Single segment case
        if boundaries.shape[0] == 0:
            return boundaries

        # Getting beg and end of previous segment
        prev_beg_seg = boundaries[0, 0].float()
        prev_end_seg = boundaries[0, 1].float()

        # Process all the segments
        for i in range(1, boundaries.shape[0]):
            beg_seg = boundaries[i, 0]
            segment_distance = beg_seg - prev_end_seg

            # Merging close segments
            if segment_distance <= close_th:
                prev_end_seg = boundaries[i, 1]

            else:
                # Appending new segments
                new_boundaries.append([prev_beg_seg, prev_end_seg])
                prev_beg_seg = beg_seg
                prev_end_seg = boundaries[i, 1]

        new_boundaries.append([prev_beg_seg, prev_end_seg])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def remove_short_segments(self, boundaries, len_th=0.250):
        """Removes segments that are too short.

        Arguments
        ---------
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.

        Returns
        -------
        new_boundaries
            The new boundaries without the short segments.
        """
        new_boundaries = []

        # Process the segments
        for i in range(boundaries.shape[0]):
            # Computing segment length
            seg_len = boundaries[i, 1] - boundaries[i, 0]

            # Accept segment only if longer than len_th
            if seg_len > len_th:
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)

        return new_boundaries

    def save_boundaries(
        self, boundaries, save_path=None, print_boundaries=True, audio_file=None
    ):
        """Saves the boundaries on a file (and/or prints them)  in a readable format.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        save_path: path
            When to store the text file containing the speech/non-speech intervals.
        print_boundaries: Bool
            Prints the speech/non-speech intervals in the standard outputs.
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio. It is used here to detect the length of the
            signal.
        """
        # Create a new file if needed
        if save_path is not None:
            f = open(save_path, mode="w", encoding="utf-8")

        # Getting the total size of the input file
        if audio_file is not None:
            sample_rate, audio_len = self._get_audio_info(audio_file)
            audio_len = audio_len / sample_rate

        # Setting the rights format for second- or sample-based boundaries
        if boundaries.dtype == torch.int:
            value_format = "% i"
        else:
            value_format = "% .2f "

        # Printing speech and non-speech intervals
        last_end = 0
        cnt_seg = 0
        for i in range(boundaries.shape[0]):
            begin_value = boundaries[i, 0]
            end_value = boundaries[i, 1]

            if last_end != begin_value:
                cnt_seg = cnt_seg + 1
                print_str = (
                    "segment_%03d " + value_format + value_format + "NON_SPEECH"
                )
                if print_boundaries:
                    print(print_str % (cnt_seg, last_end, begin_value))
                if save_path is not None:
                    f.write(print_str % (cnt_seg, last_end, begin_value) + "\n")

            cnt_seg = cnt_seg + 1
            print_str = "segment_%03d " + value_format + value_format + "SPEECH"
            if print_boundaries:
                print(print_str % (cnt_seg, begin_value, end_value))
            if save_path is not None:
                f.write(print_str % (cnt_seg, begin_value, end_value) + "\n")

            last_end = end_value

        # Managing last segment
        if audio_file is not None:
            if last_end < audio_len:
                cnt_seg = cnt_seg + 1
                print_str = (
                    "segment_%03d " + value_format + value_format + "NON_SPEECH"
                )
                if print_boundaries:
                    print(print_str % (cnt_seg, end_value, audio_len))
                if save_path is not None:
                    f.write(print_str % (cnt_seg, end_value, audio_len) + "\n")

        if save_path is not None:
            f.close()

    def energy_VAD(
        self,
        audio_file,
        boundaries,
        activation_th=0.5,
        deactivation_th=0.0,
        eps=1e-6,
    ):
        """Applies energy-based VAD within the detected speech segments.The neural
        network VAD often creates longer segments and tends to merge segments that
        are close with each other.

        The energy VAD post-processes can be useful for having a fine-grained voice
        activity detection.

        The energy VAD computes the energy within the small chunks. The energy is
        normalized within the segment to have mean 0.5 and +-0.5 of std.
        This helps to set the energy threshold.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        activation_th: float
            A new speech segment is started it the energy is above activation_th.
        deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
        eps: float
            Small constant for numerical stability.


        Returns
        -------
        new_boundaries
            The new boundaries that are post-processed by the energy VAD.
        """

        # Getting the total size of the input file
        sample_rate, audio_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the chunk length of the energy window
        chunk_len = int(self.time_resolution * sample_rate)
        new_boundaries = []

        # Processing speech segments
        for i in range(boundaries.shape[0]):
            begin_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            seg_len = end_sample - begin_sample

            # Reading the speech segment
            segment, _ = torchaudio.load(
                audio_file, frame_offset=begin_sample, num_frames=seg_len
            )

            # Create chunks
            segment_chunks = self.create_chunks(
                segment, chunk_size=chunk_len, chunk_stride=chunk_len
            )

            # Energy computation within each chunk
            energy_chunks = segment_chunks.abs().sum(-1) + eps
            energy_chunks = energy_chunks.log()

            # Energy normalization
            energy_chunks = (
                (energy_chunks - energy_chunks.mean())
                / (2 * energy_chunks.std())
            ) + 0.5
            energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

            # Apply threshold based on the energy value
            energy_vad = self.apply_threshold(
                energy_chunks,
                activation_th=activation_th,
                deactivation_th=deactivation_th,
            )

            # Get the boundaries
            energy_boundaries = self.get_boundaries(
                energy_vad, output_value="seconds"
            )

            # Get the final boundaries in the original signal
            for j in range(energy_boundaries.shape[0]):
                start_en = boundaries[i, 0] + energy_boundaries[j, 0]
                end_end = boundaries[i, 0] + energy_boundaries[j, 1]
                new_boundaries.append([start_en, end_end])

        # Convert boundaries to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def create_chunks(self, x, chunk_size=16384, chunk_stride=16384):
        """Splits the input into smaller chunks of size chunk_size with
        an overlap chunk_stride. The chunks are concatenated over
        the batch axis.

        Arguments
        ---------
        x: torch.Tensor
            Signal to split into chunks.
        chunk_size : str
            The size of each chunk.
        chunk_stride:
            The stride (hop) of each chunk.


        Returns
        -------
        x: torch.Tensor
            A new tensors with the chunks derived from the input signal.

        """
        x = x.unfold(1, chunk_size, chunk_stride)
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        return x

    def _get_audio_info(self, audio_file):
        """Returns the sample rate and the length of the input audio file"""

        # Getting the total size of the input file
        metadata = torchaudio.info(str(audio_file))
        sample_rate = metadata.sample_rate
        audio_len = metadata.num_frames
        return sample_rate, audio_len

    def upsample_VAD(self, vad_out, audio_file, time_resolution=0.01):
        """Upsamples the output of the vad to help visualization. It creates a
        signal that is 1 when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.

        Arguments
        ---------
        vad_out: torch.Tensor
            Tensor containing 1 for each frame of speech and 0 for each non-speech
            frame.
        audio_file: path
            The original audio file used to compute vad_out
        time_resolution : float
            Time resolution of the vad_out signal.

        Returns
        -------
        vad_signal
            The upsampled version of the vad_out tensor.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        beg_samp = 0
        step_size = int(time_resolution * sample_rate)
        end_samp = step_size
        index = 0

        # Initialize upsampled signal
        vad_signal = torch.zeros(1, sig_len, device=vad_out.device)

        # Upsample signal
        while end_samp < sig_len:
            vad_signal[0, beg_samp:end_samp] = vad_out[0, index, 0]
            index = index + 1
            beg_samp = beg_samp + step_size
            end_samp = beg_samp + step_size
        return vad_signal

    def upsample_boundaries(self, boundaries, audio_file):
        """Based on the input boundaries, this method creates a signal that is 1
        when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out

        Returns
        -------
        vad_signal
            The output vad signal with the same resolution of the input one.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Initialization of the output signal
        vad_signal = torch.zeros(1, sig_len, device=boundaries.device)

        # Composing the vad signal from boundaries
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            vad_signal[0, beg_sample:end_sample] = 1.0
        return vad_signal

    def double_check_speech_segments(
        self, boundaries, audio_file, speech_th=0.5
    ):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.

        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            # Read the candidate speech segment
            segment, fs = torchaudio.load(
                str(audio_file), frame_offset=beg_sample, num_frames=len_seg
            )
            speech_prob = self.get_speech_prob_chunk(segment)
            if speech_prob.mean() > speech_th:
                # Accept this as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def get_segments(
        self, boundaries, audio_file, before_margin=0.1, after_margin=0.1
    ):
        """Returns a list containing all the detected speech segments.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        before_margin: float
            Used to cut the segments samples a bit before the detected margin.
        after_margin: float
            Use to cut the segments samples a bit after the detected margin.

        Returns
        -------
        segments: list
            List containing the detected speech segments
        """
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        segments = []
        for i in range(boundaries.shape[0]):
            beg_sample = boundaries[i, 0] * sample_rate
            end_sample = boundaries[i, 1] * sample_rate

            beg_sample = int(max(0, beg_sample - before_margin * sample_rate))
            end_sample = int(
                min(sig_len, end_sample + after_margin * sample_rate)
            )

            len_seg = end_sample - beg_sample
            vad_segment, fs = torchaudio.load(
                audio_file, frame_offset=beg_sample, num_frames=len_seg
            )
            segments.append(vad_segment)
        return segments

    def get_speech_segments(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
        apply_energy_VAD=False,
        double_check=True,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.5,
        en_deactivation_th=0.0,
        speech_th=0.50,
    ):
        """Detects speech segments within the input file. The input signal can
        be both a short or a long recording. The function computes the
        posterior probabilities on large chunks (e.g, 30 sec), that are read
        sequentially (to avoid storing big signals in memory).
        Each large chunk is, in turn, split into smaller chunks (e.g, 10 seconds)
        that are processed in parallel. The pipeline for detecting the speech
        segments is the following:
            1- Compute posteriors probabilities at the frame level.
            2- Apply a threshold on the posterior probability.
            3- Derive candidate speech segments on top of that.
            4- Apply energy VAD within each candidate segment (optional).
            5- Merge segments that are too close.
            6- Remove segments that are too short.
            7- Double check speech segments (optional).


        Arguments
        ---------
        audio_file : str
            Path to audio file.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            If True, it creates overlapped small chunks (with 50% overlap).
            The probabilities of the overlapped chunks are combined using
            hamming windows.
        apply_energy_VAD: bool
            If True, a energy-based VAD is used on the detected speech segments.
            The neural network VAD often creates longer segments and tends to
            merge close segments together. The energy VAD post-processes can be
            useful for having a fine-grained voice activity detection.
            The energy thresholds is  managed by activation_th and
            deactivation_th (see below).
        double_check: bool
            If True, double checks (using the neural VAD) that the candidate
            speech segments actually contain speech. A threshold on the mean
            posterior probabilities provided by the neural network is applied
            based on the speech_th parameter (see below).
        activation_th:  float
            Threshold of the neural posteriors above which starting a speech segment.
        deactivation_th: float
            Threshold of the neural posteriors below which ending a speech segment.
        en_activation_th: float
            A new speech segment is started it the energy is above activation_th.
            This is active only if apply_energy_VAD is True.
        en_deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
            This is active only if apply_energy_VAD is True.
        speech_th: float
            Threshold on the mean posterior probability within the candidate
            speech segment. Below that threshold, the segment is re-assigned to
            a non-speech region. This is active only if double_check is True.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.

        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second of speech segments in even
            positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """

        # Fetch audio file from web if not local
        source, fl = split_path(audio_file)
        audio_file = fetch(fl, source=source)

        # Computing speech vs non speech probabilities
        prob_chunks = self.get_speech_prob_file(
            audio_file,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
            overlap_small_chunk=overlap_small_chunk,
        )

        # Apply a threshold to get candidate speech segments
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Compute the boundaries of the speech segments
        boundaries = self.get_boundaries(prob_th, output_value="seconds")

        # Apply energy-based VAD on the detected speech segments
        if apply_energy_VAD:
            boundaries = self.energy_VAD(
                audio_file,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        boundaries = self.remove_short_segments(boundaries, len_th=len_th)

        # Double check speech segments
        if double_check:
            boundaries = self.double_check_speech_segments(
                boundaries, audio_file, speech_th=speech_th
            )

        return boundaries

    def forward(self, wavs, wav_lens=None):
        """Gets frame-level speech-activity predictions"""
        return self.get_speech_prob_chunk(wavs, wav_lens)

 

def main(file,VAD):    
    
    audio_file ="VAD/Speechbrain_VAD/vad-crdnn-libriparty"
    VAD = VAD.from_hparams(source=audio_file, savedir="pretrained_models/vad-crdnn-libriparty")
    boundaries = VAD.get_speech_segments(file) #"git_file/whisper_finetune/TedTalk/commonvoice_test_tw.wav"
    print(boundaries)
    return boundaries



if __name__ =='__main__':
    main("git_file/whisper_finetune/TedTalk/commonvoice_test_tw.wav",VAD)

# if __name__ =='__main__':
#     VAD = VAD.from_hparams(source="./Speechbrain_VAD/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
#     boundaries = VAD.get_speech_segments("git_file/whisper_finetune/TedTalk/commonvoice_test_tw.wav") #"git_file/whisper_finetune/TedTalk/commonvoice_test_tw.wav"
#     print(boundaries)




