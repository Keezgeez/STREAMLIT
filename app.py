import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.scraper import search_recipes, get_recipe_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your label map — keys are ingredient names, values are integer labels
label_map = {
    "background": 0,
    "beef": 1,
    "bell_pepper": 2,
    "broccoli": 3,
    "cabbage": 4,
    "calamansi": 5,
    "carrot": 6,
    "chicken": 7,
    "eggplant": 8,
    "garlic": 9,
    "ginger": 10,
    "onions": 11,
    "onion_leaves": 12,
    "pork": 13,
    "potato": 14,
    "tofu": 15,
    "tomato": 16,
    "turmeric_powder": 17,
}

inv_label_map = {v: k for k, v in label_map.items()}

ingredient_options = list(label_map.keys())
ingredient_options.remove("background")


# --- Load your trained model once ---
@st.cache_resource(show_spinner=True)
def load_model():
    num_classes = len(label_map)  # includes background
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("latest.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


# --- Detection function ---
def detect_ingredients(image_pil, threshold=0.5):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image_pil).to(device)
    with torch.no_grad():
        outputs = model([input_tensor])

    output = outputs[0]
    detected = []
    for score, label in zip(output["scores"], output["labels"]):
        if score >= threshold:
            name = inv_label_map.get(label.item(), f"Label {label.item()}")
            detected.append((name, score.item() * 100))
    return detected


# --- Streamlit UI ---
st.title("🍲 Filipino Dish Recommender using Ingredients")

# Manual ingredient input section
st.subheader("📝 Or manually select ingredients:")
selected_manual_ingredients = st.multiselect(
    "Choose ingredients from the list:", ingredient_options
)

if st.button("🔍 Search Recipes with Selected Ingredients"):
    if selected_manual_ingredients:
        recipes = search_recipes(selected_manual_ingredients)
        if recipes:
            st.session_state.recipes = recipes
            st.session_state.selected_recipe = None
            st.session_state.recipe_steps = []
        else:
            st.warning("No recipes found. Try different ingredients.")
    else:
        st.warning("⚠️ Please select at least one ingredient.")

# Show manual recipe selector if recipes exist
if "recipes" in st.session_state and st.session_state.recipes:
    selected_recipe_title = st.selectbox(
        "Select a recipe to view instructions:",
        [r[0] for r in st.session_state.recipes],
        key="manual_recipe_select",
    )
    st.session_state.selected_recipe = next(
        url for title, url in st.session_state.recipes if title == selected_recipe_title
    )
    if st.button("Show Cooking Steps", key="manual_show_steps"):
        steps = get_recipe_steps(st.session_state.selected_recipe)
        st.session_state.recipe_steps = steps

if "recipe_steps" in st.session_state and st.session_state.recipe_steps:
    st.subheader("👨‍🍳 Step-by-step Cooking Instructions:")
    for i, step in enumerate(st.session_state.recipe_steps, 1):
        st.markdown(f"**Step {i}:** {step}")

st.markdown("---")

# Image upload / capture & detection section
option = st.radio("Choose image source:", ["Upload", "Capture (webcam)"])
image = None
if option == "Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
elif option == "Capture (webcam)":
    captured = st.camera_input("Take a picture")
    if captured:
        image = Image.open(captured).convert("RGB")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting ingredients...")
    detected_ingredients = detect_ingredients(image)

    if detected_ingredients:
        st.write("Ingredients Detected with Confidence:")
        for name, conf in detected_ingredients:
            st.markdown(f"- **{name}**: {conf:.1f}% confidence")

        detected_names = [name for name, _ in detected_ingredients]

        # Search recipes based on detected ingredients
        img_recipes = search_recipes(detected_names)
        if img_recipes:
            selected_recipe_title = st.selectbox(
                "Select a recipe to view instructions (image detection):",
                [r[0] for r in img_recipes],
                key="image_recipe_select",
            )
            img_selected_recipe = next(
                url for title, url in img_recipes if title == selected_recipe_title
            )
            if st.button("Show Cooking Steps (image)", key="image_show_steps"):
                steps = get_recipe_steps(img_selected_recipe)
                st.subheader("👨‍🍳 Step-by-step Cooking Instructions:")
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i}:** {step}")
        else:
            st.warning("No recipes found with the detected ingredients.")
    else:
        st.warning("❌ No ingredients detected in the image.")
