# captionAI
Generate image captions using open-source models. 

Using [Florence-2](https://huggingface.co/microsoft/Florence-2-large), a lightweight vision-language model by Microsoft under MIT license, for generating CAPTIONS for images in just a few seconds, integrated with FastAPI server for quick and easy deployment on cloud platforms. 

With the compression methods, such as quantization, the model performance and system requirements can be reduced drastically. From an original model footprint of `1084 MB` the model can be compressed to around `615 MB` with **4-bit** quantization. 

Quantization for the model is done using `bitesandbytes` for quick and easy implementation. 

```python
bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_compute_dtype=torch.float16,
)
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, 
    trust_remote_code=True,
).eval()
processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True,
)
 ```
<div align="center">
<img src="https://github.com/user-attachments/assets/a379833a-8c90-462e-998d-752c1d6e9f75" width=600/>
<p>Fig: Highlevel pipeline design</p>
</div>


## Setup environment
Follow the given instructions below to setup the project on local for development or to run/test the project on your environment. 

1. Clone the repository

2. Create a virtual env and install the dependencies
   ```bash
    python -m venv .venv

    # create virtual environment
    source .venv/bin/activate # linux
    .venv\Scripts\activate # windows

    # install dependencies
    pip install -r requirements.txt
   ```

3. Run the server 
   ```bash
   python app.py
   ```
   The server will download and load the model for inference and can be accessed on `localhost:8000`. 

   Head over to `localhost:8000/docs` to view the swagger UI and test out the endpoints. 


