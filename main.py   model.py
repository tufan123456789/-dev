import discord
from discord.ext import commands
from model import get_class

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)


@bot.command()
async def siniflandir(ctx):
    if ctx.message.attachments :
        # sınıflandırma işlemleri
        
        for attachment in ctx.message.attachments:
            name = attachment.filename
            url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            await ctx.send(f"Resmi şuraya kaydettiniz ./{attachment.filename}")
            
            await ctx.send("Sonucunuz yükleniyor..")
            # sınıflandırma sonucunu göstermek!!!!
            await ctx.send(get_class(model_path = "./keras_model.h5", labels_path = "./labels.txt", image_path = f"./{attachment.filename}" ))


    else:
        await ctx.send("Sınıflandırma için resim yükleyin...")


bot.run("MTIzNjM1OTcxNTQ2NzAzODcyMQ.GfNUtv.jh3OxRIxwSvjtuW0jLqNm9crNvjHepArmVv6b8")












from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np




def get_class(model_path, labels_path, image_path):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    print("model yüklenecek")
    model = load_model(model_path, compile=False)

    print("model yüklendi başarılı")
    # Load the labels
    class_names = open(labels_path, "r").readlines()
    print("labels başarılı")

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")
    print("resim yükledi")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    return class_name[2:], confidence_score




