"""Basic example information for supported plant classes.

Keys should match the lowercase class folder names in plant_dataset/raw/.
Extend only with information verified from reliable agricultural sources.
"""
PLANT_INFO = {
    "cotton": {
        "common_name": "Cotton",
        "scientific_name": "Gossypium species",
        "plant_type": "Fibre crop",
        "description": "Cotton is cultivated for its natural fibre, used in textiles.",
    },
    "maize": {
        "common_name": "Maize",
        "scientific_name": "Zea mays",
        "plant_type": "Cereal crop",
        "description": "Maize, also called corn, is grown for grain, food and animal feed.",
    },
    "rice": {
        "common_name": "Rice",
        "scientific_name": "Oryza sativa",
        "plant_type": "Cereal crop",
        "description": "Rice is a major food crop grown for its edible grains.",
    },
    "soybean": {
        "common_name": "Soybean",
        "scientific_name": "Glycine max",
        "plant_type": "Oilseed and pulse crop",
        "description": "Soybean is a legume grown for its protein-rich seeds and oil.",
    },
    "tomato": {
        "common_name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "plant_type": "Vegetable crop",
        "description": "Tomato is cultivated for its edible fruits.",
    },
}
