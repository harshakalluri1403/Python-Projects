import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the image captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

# Load pre-trained ResNet model
resnet = resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
for param in resnet.parameters():
    param.requires_grad = False

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the image captioning model parameters
embed_size = 256
hidden_size = 512
vocab_size = 10000  # Change based on your vocabulary size
num_layers = 1

# Initialize the image captioning model
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)

# Load pre-trained weights (you can train the model on your dataset)
# model.load_state_dict(torch.load('image_captioning_model.pth'))
model.eval()

# Load the vocabulary
with open('vocab.txt', 'r') as f:
    vocab = f.read().splitlines()

# Remove stopwords from the vocabulary
stop_words = set(stopwords.words('english'))
vocab = [word for word in vocab if word not in stop_words]

# Function to preprocess and extract image features
def get_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image_features = resnet(Variable(image)).squeeze(2).squeeze(2)
    return image_features

# Function to generate captions for the given image
def generate_caption(image_path, max_len=20):
    image_features = get_image_features(image_path)
    sampled_ids = []
    inputs = torch.LongTensor([vocab.index('<start>')])

    for i in range(max_len):
        outputs = model(image_features, inputs)
        _, predicted = outputs.max(2)
        sampled_ids.append(predicted.item())
        inputs = predicted.squeeze(1)

        if sampled_ids[-1] == vocab.index('<end>'):
            break

    caption = [vocab[i] for i in sampled_ids]
    return ' '.join(caption[1:-1])  # Exclude <start> and <end>

# Test the image captioning model
image_path = 'example.jpg'  # Replace with your image path
caption = generate_caption(image_path)
print(f"Image Caption: {caption}")

# Display the image
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()
