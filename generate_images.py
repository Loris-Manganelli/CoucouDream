
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
import ip_adapter
import os
import random
import matplotlib.pyplot as plt
from matplotlib.image import imread
import joblib
import time



class Run :

    def __init__(self, is_double, num_images_per_class, dataset_name, n_shots):
        self.is_double = is_double
        self.num_images_per_class = num_images_per_class
        self.dataset_name = dataset_name
        self.n_shots = n_shots
        self.dataset_path = './DataDream/data/'+self.dataset_name+'/real_train_fewshot/seed0'

        vicky, dirs, nicolas = next(os.walk(self.dataset_path))
        self.n_class = len(dirs)


    def load_models(self):

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            image_encoder=image_encoder
        )
        

        self.pipeline.load_ip_adapter("h94/IP-Adapter",
        subfolder="models",
        weight_name=["ip-adapter-plus_sd15.safetensors"]*(self.is_double + 1)
        )

        if self.is_double :
            self.pipeline.set_ip_adapter_scale([0.5,0.5]) 
        else:
            self.pipeline.set_ip_adapter_scale([1.0])

        self.pipeline.enable_model_cpu_offload()




    def generate_images(self):

        n = 0
        num_images = 2 if self.is_double else 1
        times = []

        for root, dirs, files in os.walk(self.dataset_path):
            for dir_name in dirs:

                dir_path = os.path.join(root, dir_name)           
                image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][:self.n_shots]
                

                for k in range(self.num_images_per_class):
                    
                    t0 = time.time()
                    selected_images = []

                    if self.is_double:

                        selected_files = random.sample(image_files, num_images)
                        image = Image.open(os.path.join(dir_path, selected_files[0]))
                        image2 = Image.open(os.path.join(dir_path, selected_files[1]))

                        ipadapter_prompt = [image, image2]

                    else :

                        selected_files = random.sample(image_files, num_images)
                        image = Image.open(os.path.join(dir_path, selected_files[0]))

                        ipadapter_prompt = image

                    generator = torch.Generator(device="cuda").manual_seed(0)

                    image = self.pipeline(
                        prompt="A satellite photo of " + dir_name,
                        ip_adapter_image=ipadapter_prompt,
                        # negative_prompt="lowres, worst quality, low quality",
                        num_inference_steps=50,
                        num_images_per_prompt=1,
                        generator=generator,
                        guidance_scale=4.0,
                        seed = random.randint(0,50000),

                    ).images[0]

                    str_ipa = "single" if self.is_double else "double"
                    directory = 'double-ipa-outputs/' + self.dataset_name + '_' + str_ipa +'/' + dir_name

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    image.save( directory + '/image' + str(n) + '.png' )
                    n+=1
                        
                    times.append(time.time()-t0)
            
        return sum(times)/len(times)






if __name__ == '__main__' :

    times = {}
    for double in [True, False]:

        run = Run(is_double=double, num_images_per_class=200, dataset_name='eurosat', n_shots=16)
        run.load_models()
        time = run.generate_images()
        times[double] = time

    joblib.dump(times, 'output.joblib')
