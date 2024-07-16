# captionAI
Generate image captions using open-source models

Using `Florence-2`, a lightweight vision-language model by Microsoft under MIT license, for generating CAPTIONS for images in just a few seconds, integrated with FastAPI server for quick and easy deployment on cloud platforms. 

With the compression methods, such as quantization, the model performance and system requirements can be reduced drastically. From an original model footprint of `1084 MB` the model can be compressed to around `615 MB` with **4-bit** quantization. 

Quantization for the model done using `bitesandbytes` Configuration for quick and easy implementation. 

```python
bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_compute_dtype=torch.float16,
)
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
 ```
