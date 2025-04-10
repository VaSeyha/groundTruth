import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import itertools

# --------------------------- Streamlit UI ---------------------------
st.title("🖊️ Hand-drawn Annotation Tool")

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

# ---------------------- Coordinate Processing ----------------------

def extract_coordinates(json_data):
    nested_coords = []
    temp_list = []

    if json_data is not None:
        objects = json_data.get("objects", [])
        for obj in objects:
            if obj.get("type") == "path" and "path" in obj:
                for point in obj["path"]:
                    if isinstance(point, list):
                        cmd = point[0]
                        coords = point[1:]

                        # Process coordinates in pairs (x, y)
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x, y = coords[i], coords[i + 1]
                                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                    temp_list.extend([x, y])
        
        while len(temp_list) >= 16:
            nested_coords.append(temp_list[:16])
            temp_list = temp_list[16:]

        
        nested_coords.append(temp_list)

    return nested_coords

def scale_coordinates(nested_coords):
    flat = list(itertools.chain(*nested_coords))
    x_vals = flat[::2]
    y_vals = flat[1::2]

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    norm_x = [round((x - min_x) / x_range, 8) for x in x_vals]
    norm_y = [round((y - min_y) / y_range, 8) for y in y_vals]

    normalized = list(itertools.chain(*zip(norm_x, norm_y)))

    normalized_data = []
    while len(normalized) >= 16:
        normalized_data.append(normalized[:16])
        normalized = normalized[16:]
    
    if normalized:
        normalized.extend([0] * (16 - len(normalized)))
        nested_coords.append(normalized)

    return normalized_data

# --------------------------- Model Code ----------------------------

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_out):
        weights = torch.softmax(self.attn(gru_out), dim=1)
        return (gru_out * weights).sum(dim=1)

class DeepGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size * 2)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_out = self.attention(gru_out)
        x = self.layer_norm(attention_out)

        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))
        return self.fc3(x)

# ------------------------ Model Parameters -------------------------

input_size = 16
hidden_size = 256
num_layers = 4
output_size = 10
dropout = 0.3

model = DeepGRUModel(input_size, hidden_size, num_layers, output_size, dropout)
model.load_state_dict(torch.load("best_gru_model_VF.pth", map_location=torch.device('cpu')))
model.eval()

# ----------------------------- Main -------------------------------

nested_coords = extract_coordinates(canvas_result.json_data)

if nested_coords:
    normalized_data = scale_coordinates(nested_coords)
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
    tensor_data = torch.stack([tensor_data])

    with torch.no_grad():
        output = model(tensor_data)
        prediction = torch.argmax(output, dim=1).cpu().numpy()

    # st.subheader("Model Output")
    # st.write("Raw Scores (logits):", output)
    st.write("Prediction:", prediction)
else:
    st.write("")
    # st.warning("")

