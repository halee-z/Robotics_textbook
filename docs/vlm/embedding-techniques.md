---
sidebar_position: 3
---

# Embedding Techniques for Robotics Applications

## Overview

Embedding techniques are fundamental to Vision-Language Models in robotics, enabling the conversion of high-dimensional sensory data into meaningful representations that can guide robot behavior. This section covers the key embedding methods that enable robots to understand and interact with their environment.

## Visual Embeddings

### Feature Extraction Methods

#### Convolutional Neural Networks (CNNs)
Traditional CNNs extract hierarchical visual features from images:

```python
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Pre-trained ResNet for feature extraction
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, output_dim)  # ResNet50 outputs 2048-dim features
    
    def forward(self, images):
        # Extract features
        features = self.backbone(images)
        features = torch.flatten(features, 1)
        
        # Project to desired dimension
        embeddings = self.projection(features)
        
        return embeddings
```

#### Vision Transformers (ViTs)
ViTs process images as sequences of patches, often providing better representations:

```python
from transformers import ViTModel, ViTConfig

class ViTEmbedder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config)
        self.projection = nn.Linear(self.vit.config.hidden_size, output_dim)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        # Use the pooled output (cls token)
        sequence_output = outputs.pooler_output
        embeddings = self.projection(sequence_output)
        return embeddings
```

### Multi-Scale Embeddings

Robots often need to process visual information at multiple scales:

```python
class MultiScaleVisualEmbedder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Different scales for different purposes
        self.fine_grained = ViTEmbedder(output_dim=output_dim//2)  # For object details
        self.contextual = CNNEncoder(output_dim=output_dim//2)    # For scene context
    
    def forward(self, image, region_of_interest=None):
        # Context embedding from full image
        context_embedding = self.contextual(image)
        
        # Fine-grained embedding from region of interest or full image
        if region_of_interest is not None:
            # Crop and process specific region
            cropped_image = self.crop_image(image, region_of_interest)
            fine_embedding = self.fine_grained(cropped_image)
        else:
            fine_embedding = self.fine_grained(image)
        
        # Combine embeddings
        combined_embedding = torch.cat([context_embedding, fine_embedding], dim=-1)
        
        return combined_embedding
    
    def crop_image(self, image, bbox):
        # Implementation for cropping image based on bounding box
        # bbox format: [x1, y1, x2, y2]
        pass
```

## Language Embeddings

### Pre-trained Language Models

#### BERT-based Embeddings
BERT provides contextual language understanding:

```python
from transformers import BertModel, BertTokenizer

class BERTLanguageEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding for the entire sentence
            embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return embedding
```

#### Sentence Transformers
For robotics applications requiring semantic similarity:

```python
from sentence_transformers import SentenceTransformer

class RoboticLanguageEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode_sentences(self, sentences):
        """Encode multiple sentences into embeddings"""
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings
    
    def compute_similarity(self, sentence1, sentence2):
        """Compute semantic similarity between two sentences"""
        emb1 = self.encode_sentences([sentence1])
        emb2 = self.encode_sentences([sentence2])
        similarity = torch.cosine_similarity(emb1, emb2, dim=1)
        return similarity.item()
```

## Cross-Modal Embeddings

### CLIP Embeddings

CLIP creates a shared embedding space for visual and textual information:

```python
import clip
import torch
from PIL import Image

class CLIPEmbedder:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
    
    def embed_image(self, image_path):
        """Embed an image into the CLIP space"""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # Normalize embeddings
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu()
    
    def embed_text(self, text):
        """Embed text into the CLIP space"""
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize embeddings
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu()
    
    def compute_similarity(self, image_path, text):
        """Compute similarity between image and text"""
        image_features = self.embed_image(image_path)
        text_features = self.embed_text(text)
        
        similarity = (image_features @ text_features.T).item()
        return similarity
```

### Robot-Specific Embeddings

Creating embeddings that are specifically optimized for robotics tasks:

```python
class RoboticActionEmbedder(nn.Module):
    def __init__(self, vocab_size, action_space_dim, embedding_dim=512):
        super().__init__()
        
        # Embed language instructions
        self.lang_embedder = nn.Embedding(vocab_size, embedding_dim)
        
        # Embed visual scene context
        self.visual_encoder = ViTEmbedder(output_dim=embedding_dim)
        
        # Embed current robot state
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 128),  # Position and orientation
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim//2)
        )
        
        # Fuse all modalities
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2 + embedding_dim//2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Map to action space
        self.action_head = nn.Linear(embedding_dim, action_space_dim)
    
    def forward(self, instruction_ids, visual_input, robot_state):
        # Embed language instruction
        lang_embedding = self.lang_embedder(instruction_ids).mean(dim=1)  # Average over sequence
        
        # Embed visual input
        visual_embedding = self.visual_encoder(visual_input)
        
        # Embed robot state
        state_embedding = self.state_encoder(robot_state)
        
        # Concatenate all embeddings
        fused_input = torch.cat([lang_embedding, visual_embedding, state_embedding], dim=-1)
        
        # Fuse and generate action embedding
        fused_embedding = self.fusion(fused_input)
        
        # Generate action
        action = self.action_head(fused_embedding)
        
        return action, fused_embedding
```

## Embedding Alignment Techniques

### Contrastive Learning

Align visual and language embeddings using contrastive loss:

```python
class ContrastiveAligner(nn.Module):
    def __init__(self, embedding_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        # Visual and text encoders
        self.visual_encoder = ViTEmbedder(output_dim=embedding_dim)
        self.text_encoder = RoboticLanguageEncoder()  # Simplified
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, texts):
        # Encode images
        image_features = self.visual_encoder(images)
        image_features = F.normalize(image_features, dim=1)
        
        # Encode texts
        text_features = self.text_encoder.encode_texts(texts)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        # Compute contrastive loss
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss, image_features, text_features
```

### Triplet Embeddings

Using triplet loss for improved embedding quality:

```python
class TripletEmbedder(nn.Module):
    def __init__(self, embedding_dim=512, margin=0.2):
        super().__init__()
        self.margin = margin
        self.visual_encoder = ViTEmbedder(output_dim=embedding_dim)
        self.text_encoder = BERTLanguageEmbedder()
        
    def forward(self, anchor_images, positive_texts, negative_texts):
        # Encode all inputs
        anchor_embeddings = self.visual_encoder(anchor_images)
        pos_text_embeddings = self.text_encoder.embed_text(positive_texts)
        neg_text_embeddings = self.text_encoder.embed_text(negative_texts)
        
        # Compute triplet loss
        pos_distance = F.pairwise_distance(anchor_embeddings, pos_text_embeddings)
        neg_distance = F.pairwise_distance(anchor_embeddings, neg_text_embeddings)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        loss = loss.mean()
        
        return loss
```

## Embedding Optimization for Robotics

### Memory-Efficient Embeddings

For resource-constrained robotics platforms:

```python
class QuantizedRoboticEmbedder:
    def __init__(self, full_model, bits=8):
        self.full_model = full_model
        self.bits = bits
        self.scale = None
        self.zero_point = None
    
    def quantize(self, example_input):
        """Quantize the model for efficient inference"""
        # Forward pass to get example outputs
        with torch.no_grad():
            full_output = self.full_model(example_input)
        
        # Compute quantization parameters
        self.scale = (full_output.max() - full_output.min()) / (2**self.bits - 1)
        self.zero_point = -full_output.min() / self.scale
        
        # Create quantized version of the model
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.full_model, {nn.Linear}, dtype=torch.qint8
        )
    
    def embed(self, input_data):
        """Embed using quantized model"""
        with torch.no_grad():
            return self.quantized_model(input_data)
```

### Continual Embedding Updates

For robots that continuously learn new concepts:

```python
class ContinualEmbedder:
    def __init__(self, base_embedding_model):
        self.model = base_embedding_model
        self.exemplars = {}  # Store exemplars for replay
        self.classifier = nn.Linear(512, 10)  # Initial 10 classes
        self.new_class_idx = 10  # Index for new classes
    
    def add_new_concept(self, new_data, new_labels):
        """Add new concepts while preventing forgetting"""
        # Store exemplars of old concepts
        with torch.no_grad():
            old_embeddings = self.model.embed(new_data[:len(new_data)//2])
            for i, emb in enumerate(old_embeddings):
                class_id = new_labels[i]
                if class_id not in self.exemplars:
                    self.exemplars[class_id] = []
                self.exemplars[class_id].append(emb.cpu())
        
        # Expand classifier for new classes
        self.expand_classifier(len(set(new_labels)))
        
        # Fine-tune with replay of old concepts
        self.finetune_with_replay(new_data, new_labels)
    
    def expand_classifier(self, num_new_classes):
        """Expand the classifier for new classes"""
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        
        in_features = self.classifier.in_features
        old_out_features = self.classifier.out_features
        new_out_features = old_out_features + num_new_classes
        
        # Create new classifier
        new_classifier = nn.Linear(in_features, new_out_features)
        
        # Copy old weights and initialize new weights
        with torch.no_grad():
            new_classifier.weight[:old_out_features] = old_weight
            new_classifier.bias[:old_out_features] = old_bias
        
        self.classifier = new_classifier
    
    def finetune_with_replay(self, new_data, new_labels):
        """Fine-tune with replay of old exemplars"""
        optimizer = torch.optim.Adam(list(self.model.parameters()) + 
                                   list(self.classifier.parameters()))
        
        # Combine new data with exemplars from memory
        all_data = new_data
        all_labels = new_labels
        
        for class_id, exemplars in self.exemplars.items():
            # Add exemplars with their old labels
            all_data.extend(exemplars)
            all_labels.extend([class_id] * len(exemplars))
        
        # Training loop
        for epoch in range(5):  # Few epochs to avoid catastrophic forgetting
            for batch_data, batch_labels in self.create_batches(all_data, all_labels):
                optimizer.zero_grad()
                embeddings = self.model.embed(batch_data)
                outputs = self.classifier(embeddings)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                optimizer.step()
```

## Hands-on Exercise

1. Implement a CLIP-based embedder that can match robotic action descriptions to relevant images in a dataset.

2. Design an embedding technique that combines visual, language, and robot state information for decision making.

3. Consider how you would optimize these embeddings for real-time robotics applications with limited computational resources.

The next section will explore how these embeddings are used for planning with Vision-Language Models in robotics.