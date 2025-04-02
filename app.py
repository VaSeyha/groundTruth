import streamlit as st
from streamlit_drawable_canvas import st_canvas
import itertools

st.title("Hand-drawn Annotation Tool")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=3,
    stroke_color="black",
    background_color="white",
    height=400,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)


import torch
# The align_shape function
def align_shape(batch):
    coordinates = batch  # batch is now just a list of coordinates (or a single item)
    
    # Find the max sequence length in the batch
    max_len = max(coor.shape[0] for coor in coordinates)
    
    # Pad tensors to have the same shape
    padded_coordinates = [
        torch.cat((coor, torch.zeros(max_len - coor.shape[0], coor.shape[1])), dim=0) for coor in coordinates
    ]
    
    return torch.stack(padded_coordinates)








nested_coordinates = []  # Variable to store the nested list of coordinates

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])

    if objects:
        temp_list = []  # Temporary list to store x, y pairs
        for obj in objects:
            if obj.get("type") == "path" and "path" in obj:
                for point in obj["path"]:
                    if isinstance(point, list) and len(point) >= 3:
                        _, x, y = point[:3]  # Extract x, y from the path
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            temp_list.extend([x, y])  # Append x, y in sequence

        # Now, split the temp_list into chunks of 16 values each
        while len(temp_list) >= 16:
            nested_coordinates.append(temp_list[:16])  # Add the first 16 elements
            temp_list = temp_list[16:]  # Remove the first 16 elements

        # If there are any remaining coordinates, pad with zeros until the length is 16
        if temp_list:
            temp_list.extend([0] * (16 - len(temp_list)))  # Pad with zeros
            nested_coordinates.append(temp_list)  # Add the padded list

        # If there are still not enough lists to complete 16 lists, pad with empty lists
        while len(nested_coordinates) < 16:
            nested_coordinates.append([0] * 16)  # Add empty lists of 16 zeros


def scale_coordinate(corr):
    
    # Flatten the list using itertools.chain()
    flat_list = list(itertools.chain(*corr))

    # Extract x and y values using slicing
    x_values = flat_list[::2]  # x values are at even indices
    y_values = flat_list[1::2]  # y values are at odd indices

    # Find min and max
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    x_range = max_x - min_x
    y_range = max_y - min_y
    # Apply min-max scaling with 8 decimal places
    normalized_x = [round((x - min_x) / x_range, 8) for x in x_values]
    normalized_y = [round((y - min_y) / y_range, 8) for y in y_values]


    # Reconstruct the normalized data in the original nested list format
    normalized_flat = list(itertools.chain(*zip(normalized_x, normalized_y)))

    #reconstruct the original data format
    normalized_data = []
    index = 0
    for row in corr:
        length = len(row)  # Get the original row length
        normalized_data.append(normalized_flat[index:index+length])
        index += length  # Move to the next chunk

    return normalized_data

normalized_data = scale_coordinate(nested_coordinates)



# Convert the coordinates to a tensor
coordinates_tensor = torch.tensor(normalized_data, dtype=torch.float32)

# # Apply the align_shape function directly to your data
# aligned_coordinates = align_shape([coordinates_tensor])  # Pass the list of tensors
coordinates_tensor = torch.stack([coordinates_tensor])

import torch
import torch.nn as nn

# 🔹 BLSTM Model with Attention, Layer Norm, and Residual Connections
class AdvancedBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(AdvancedBLSTM, self).__init__()
        
        # Bidirectional LSTM
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention Mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation & Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        blstm_out, _ = self.blstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # Apply Layer Normalization
        blstm_out = self.layer_norm(blstm_out)
        
        # Attention Weights Calculation
        attn_weights = torch.softmax(self.attention(blstm_out), dim=1)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attn_weights * blstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Residual Connection: Adding LSTM Output to Context Vector
        combined = blstm_out[:, -1, :] + context_vector  # (batch_size, hidden_size*2)
        
        # Fully Connected Layers
        x = self.relu(self.fc1(self.dropout(combined)))
        x = self.relu(self.fc2(self.dropout(x)))
        out = self.fc3(x)

        return out
    

# Model Hyperparameters
input_size = 16
hidden_size = 256
num_layers = 4
output_size = 10
dropout_rate = 0.3
batch_size = 32

model = AdvancedBLSTM(input_size, hidden_size, num_layers, output_size, dropout_rate)  

# Load the saved state_dict
model.load_state_dict(torch.load("best_blstm_model.pth"))

# Set the model to evaluation mode
model.eval()

# Get predictions
with torch.no_grad():  # Disable gradient computation for inference
    output = model(coordinates_tensor)
    preds = torch.argmax(output, dim=1)
    answer = preds.cpu().numpy()

# print("Predicted Output:", output)


if nested_coordinates:
    st.write("Nested List of Coordinates:")
    # st.write(nested_coordinates)  # Display the nested list
    # st.write(normalized_data)
    # st.write(aligned_coordinates)
    st.write(output) # Write
    st.write(preds)
    st.write(answer)
else:
    st.warning("No valid coordinates detected. Please draw something.")
