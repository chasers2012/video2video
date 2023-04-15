# Author: Filarius
# Credits: orcist1
# https://github.com/Filarius
# https://github.com/Filarius/video2video

import os,sys
from subprocess import Popen, PIPE
import numpy as np
from PIL import Image
from random import randint
import platform
import modules
from modules.shared import state
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser,remove_current_script_callbacks
import math
import torch
import gradio as gr
from modules import scripts, deepbooru
from modules.images import save_image
from modules.sd_samplers import sample_to_image
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.processing import Processed, process_images, StableDiffusionProcessingImg2Img
from modules import processing
from modules.shared import state
import modules.shared as shared
import ffmpeg as libffmpeg

try: # make me know if there is better solution
    import skvideo
except:
    if 'VENV_DIR' in os.environ:
        path = os.path.join(os.environ["VENV_DIR"],'scripts','python')
    elif 'PYTHON' in os.environ:
        path = os.path.join(os.environ["PYTHON"], 'python')
    else:
        from shutil import which
        path = which("python")
    os.system(path+ " -m pip install sk-video")
    import skvideo

class LatentMemory:
    def __init__(self, interp_factor_start=0.1, interp_factor_end = 0.95):
        self.latents_now = []
        self.latents_mem = []
        self.flushed = False
        self.interp_factor_start = interp_factor_start #
        self.interp_factor_end = interp_factor_end

    def put(self,latent):
        self.latents_now.append(latent)

    def get(self):
        lm = self.latents_mem.pop(0)
        if len(self.latents_mem)==0:
            self.flushed = False
        return lm

    def interpolate(self, latent1, latent2, step_factor):
        factor = self.interp_factor_start + step_factor * (self.interp_factor_end-self.interp_factor_start)
        latent = latent1 * (1. - factor) + latent2 * factor
        return latent

    def flush(self):
        self.latents_mem = self.latents_now
        self.latents_now = []
        self.flushed = True
        
    def discard(self):
        self.latents_mem = []
        self.latents_now = []
        self.flushed = False


class Script(scripts.Script):
    # script title to show in ui
    def title(self):
        return 'Video2Video'

    def show(self, is_img2img):
        #return scripts.AlwaysVisible
        return is_img2img

    def __init__(self):
        self.img2img_component = gr.Image()
        self.img2img_inpaint_component = gr.Image()
        self.is_have_callback = False
        #from RIFE_HDv3 import Model
        #model = Model()
        #model.load_model('flownet.pkl', -1)

    def interrogate(self, image, model):
        if model is None or image is None:
            return ''
        
        # Override object param
        if model == "clip":
            processed = shared.interrogator.interrogate(image)
        elif model == "clip(no_style)":
            processed = shared.interrogator.interrogate(image).split(',')[0]
        elif model == "deepdanbooru":
            processed = deepbooru.model.tag(image)
        else:
            return ''

        return processed


    # ui components
    def ui(self, is_visible):
            def img_dummy_update(arg):
                if arg is None:
                    dummy_image = "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"
                    import io
                    import base64
                    dummy_image = io.BytesIO(base64.b64decode(dummy_image))
                    img = Image.open(dummy_image)
                    return img
                else:
                    return arg
            with gr.Row():
                file = gr.File(label="Upload Video", file_types = ['.*;'], live=True, file_count = "single")
                tmp_path = gr.Textbox(label='Or path to file', lines=1, value='')
            with gr.Row():
                fps = gr.Slider(
                    label="FPS change",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=24,
                )
            with gr.Row():
                gr.HTML(value='<div class="text-center block">Latent space temporal blending</div></br>')
            with gr.Row():
                with gr.Column(min_width=100):
                    freeze_input_fps = gr.Checkbox(label='Keep input frames', value=False)
                    keep_fps = gr.Checkbox(label='Keep FPS', value=False)
                    use_controlnet = gr.Checkbox(label='Using ControlNet', value=False)
                with gr.Column(min_width=100):
                        interp_factor_start  = gr.Slider(
                            label="Strength Start",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.2,
                        )
                        interp_factor_end = gr.Slider(
                            label="Strength End",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.2,
                        )
            with gr.Row():
                mae_threshold = gr.Slider(
                    label="MAE Threshold",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.15,
                )
                
            with gr.Row():
                interrogator=gr.Dropdown(value='none', choices=['none','clip', 'clip(no_style)', 'deepdanbooru'],label="Select interrogator:")
                
            with gr.Row():
                resume_file = gr.File(label="Upload Video to Resume", file_types = ['.*;'], live=True, file_count = "single")
                
            file.upload(fn=img_dummy_update,inputs=[self.img2img_component],outputs=[self.img2img_component])
            tmp_path.change(fn=img_dummy_update,inputs=[self.img2img_component],outputs=[self.img2img_component])
            #self.img2img_component.update(Image.new("RGB",(512,512),0))
            return [tmp_path, fps, file,interp_factor_start,interp_factor_end, mae_threshold, freeze_input_fps,keep_fps,use_controlnet, interrogator, resume_file]

    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_image":
            self.img2img_component = component
            return self.img2img_component

    def run(self, p:StableDiffusionProcessingImg2Img, file_path, fps, file_obj, interp_factor_start, interp_factor_end, mae_threshold, freeze_input_fps, keep_fps, use_controlnet, interrogator, resume_file_obj, *args):
            save_dir = "outputs/img2img-video"
            os.makedirs(save_dir, exist_ok=True)
            path = modules.paths.script_path
            if platform.system() == 'Windows':
                ffmpeg.install(path)
                import skvideo
                skvideo.setFFmpegPath(os.path.join(path, "ffmpeg"))
            import skvideo.io

            self.latentmem = LatentMemory(interp_factor_start=interp_factor_start, interp_factor_end=interp_factor_end)
            if not self.is_have_callback:
                def callback(params: CFGDenoiserParams):
                    self.latentmem.put(params.x)
                    if self.latentmem.flushed:
                        latent = self.latentmem.get()                        
                        if params.sampling_step < params.total_sampling_steps -2:
                            params.x = self.latentmem.interpolate(params.x, latent, params.sampling_step / params.total_sampling_steps)

                    if params.sampling_step == params.total_sampling_steps - 2:
                        self.latentmem.flush()

                on_cfg_denoiser(callback)
                self.is_have_callback = True

            if not os.path.isfile(file_path):
                file_path = file_obj.name
                
            resume_file_path = resume_file_obj.name if resume_file_obj is not None else None
            initial_seed = p.seed
            if initial_seed == -1:
                initial_seed = randint(100000000,999999999)
                p.seed = initial_seed
            processing.fix_seed(p) 

            p.do_not_save_grid = True
            p.do_not_save_samples = False
            p.batch_count = 1

            start_time = "00:00:00"
            end_time = "00:00:00"
            start_time = start_time.strip()
            end_time = end_time.strip()

            if start_time == "" or start_time == "hh:mm:ss":
                start_time = "00:00:00"
            if end_time == "00:00:00" or end_time == "hh:mm:ss":
                end_time = ""

            time_interval = (
                f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
            )

            input_file = os.path.normpath(file_path.strip())
            resume_file = os.path.normpath(resume_file_path.strip()) if resume_file_path is not None else None

            if freeze_input_fps:
                decoder = skvideo.io.FFmpegReader(input_file)
                resume_decoder = skvideo.io.FFmpegReader(resume_file) if resume_file is not None else None
            else:
                decoder = skvideo.io.FFmpegReader(input_file,outputdict={
                    '-r':str(fps)
                })
                resume_decoder = skvideo.io.FFmpegReader(resume_file,outputdict={
                    '-r':str(fps)
                }) if resume_file is not None else None
                
            
            
            state.job_count = decoder.inputframenum
            state.job_no = 0

            output_file = os.path.basename(input_file)
            output_file = os.path.splitext(output_file)[0]
            
            i=1
            while os.path.isfile(f'{save_dir}/{output_file}_{i}.mp4'):
                i+=1
            output_file_final = f'{save_dir}/{output_file}_{i}.mp4'
            output_file = f'{save_dir}/{output_file}_{i}-noaudio.mp4'

            if keep_fps:
                if '@r_frame_rate' in decoder.probeInfo['video']:
                    fps = decoder.probeInfo['video']['@r_frame_rate']
                elif '@avg_frame_rate' in decoder.probeInfo['video']:
                    fps = decoder.probeInfo['video']['@avg_frame_rate']
                else:
                    print('===================================================')
                    print("[Vid2Vid] Source FPS not known to keep original FPS")
            if isinstance(fps,str):
                sp = fps.split('/')
                if len(sp)==2 and (sp[1]=='1'):
                    fps = sp[0]
            print(output_file)
            encoder = skvideo.io.FFmpegWriter(output_file,
                inputdict={
                    '-r': str(fps),
                },
                outputdict={
                #'-r':str(fps),
                '-vcodec': 'libx264',
                '-crf': '22'
            })

            batch = []
            is_last = False
            frame_generator = decoder.nextFrame()
            resume_frame_generator = resume_decoder.nextFrame() if resume_file is not None else None
            original_prompt = p.prompt
            prev_image = None
            while not is_last:
                if state.interrupted:
                    break
                extra_prompts = []
                prompts = None
                raw_image = next(frame_generator,[])
                resume_frame = next(resume_frame_generator,[]) if resume_frame_generator is not None else []

                image_PIL = None
                if len(raw_image)==0:
                    is_last = True
                else:
                    image_PIL = Image.fromarray(raw_image,mode='RGB')
                    if prev_image is not None:
                        curr_image = np.array(image_PIL).ravel()
                        MAE = np.sum(np.abs(np.subtract(curr_image,prev_image,dtype=np.float))) / curr_image.shape[0] / 255
                        print()
                        print("MAE:")
                        print(MAE)
                        if MAE > mae_threshold:
                            self.latentmem.discard()
                    prev_image=np.array(image_PIL).ravel()

                    if len(resume_frame) > 0:
                        encoder.writeFrame(np.asarray(Image.fromarray(resume_frame, mode='RGB')).copy())
                        state.job_no += 1
                        continue
                    prompts = self.interrogate(image_PIL, interrogator)
                    extra_prompts.append(prompts)
                    batch.append(image_PIL)

                if (len(batch) == p.batch_size) or ( (len(batch) > 0) and is_last ):
                    p.seed = initial_seed
                    p.init_images = batch
                    
                    p.prompt = [original_prompt + ', ' + x  for x in extra_prompts]
                    batch = []
                    try:
                        proc = process_images(p)
                    except:
                        break
                    
                    controlnet_counter = 0
                    for output in proc.images:
                        if use_controlnet:
                            if controlnet_counter == 1:
                                continue
                        
                        if isinstance(output, (np.ndarray, np.generic) ):
                            newoutput = Image.fromarray(output,mode='RGB')
                        else:
                            newoutput = output

                        if newoutput.mode != "RGB":
                            newoutput = newoutput.convert("RGB")
                            
                        encoder.writeFrame(np.asarray(newoutput).copy())
                        state.job_no += 1

                        controlnet_counter = 1
                # put original prompt back incase other plugins change it during process
                p.prompt=original_prompt
            #encoder.write_eof()
            encoder.close()
            ffmpeg.concat_audio(input_file, output_file, output_file_final, path=path if platform.system() == 'Windows' else None )
            
            remove_current_script_callbacks()
            self.is_have_callback = False
            return Processed(p, [], p.seed, proc.info)


class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        try:
            print(self._cmdln)
            self._process = Popen(
                self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
            )
        except Exception as e:
            print(e)

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def install(path):
        from basicsr.utils.download_util import load_file_from_url
        from zipfile import ZipFile

        ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip"
        ffmpeg_dir = os.path.join(path, "ffmpeg")

        if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, "ffmpeg.exe"))):
            print("Downloading FFmpeg")
            ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)
            with ZipFile(ckpt_path, "r") as zipObj:
                listOfFileNames = zipObj.namelist()
                for fileName in listOfFileNames:
                    if "/bin/" in fileName:
                        zipObj.extract(fileName, ffmpeg_dir)
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffmpeg.exe"),
                os.path.join(ffmpeg_dir, "ffmpeg.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffplay.exe"),
                os.path.join(ffmpeg_dir, "ffplay.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffprobe.exe"),
                os.path.join(ffmpeg_dir, "ffprobe.exe"),
            )

            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin"))
            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
            print("Downloading FFmpeg: Done")
        return

    @staticmethod
    def concat_audio(src, dst, out, path=None):
        src=os.path.abspath(src)
        dst=os.path.abspath(dst)
        input_video = libffmpeg.input(dst)
        input_audio = libffmpeg.input(src)
        cmd = os.path.abspath(os.path.join(os.path.join(path, "ffmpeg"), "ffmpeg.exe")) if path is not None else None
        libffmpeg.concat(input_video, input_audio, v=1, a=1).output(out).run(cmd=cmd)

        
    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds



