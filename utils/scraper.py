import requests
from bs4 import BeautifulSoup


def search_recipes(ingredients):
    recipes = []
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Attempt to search with all ingredients
    search_url = f"https://panlasangpinoy.com/?s=" + "+".join(ingredients)
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for card in soup.select("h2.entry-title a"):
        title = card.text.strip()
        url = card["href"]
        recipes.append((title, url))

    # If no recipes found, try searching with individual ingredients
    if not recipes:
        for ingredient in ingredients:
            search_url = f"https://panlasangpinoy.com/?s={ingredient}"
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            for card in soup.select("h2.entry-title a"):
                title = card.text.strip()
                url = card["href"]
                if (title, url) not in recipes:
                    recipes.append((title, url))

    return recipes

def get_recipe_steps(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise error for bad status codes
    except requests.RequestException as e:
        print(f"❌ Error fetching the recipe page: {e}")
        return ["Failed to retrieve recipe."]

    soup = BeautifulSoup(response.text, "html.parser")

    steps = []

    # Try WPRM structured instructions first
    instructions = soup.select("div.wprm-recipe-instruction-text")
    if instructions:
        print("✅ Found structured recipe steps.")
        for step in instructions:
            step_text = step.get_text(strip=True)
            if step_text:
                steps.append(step_text)
    else:
        # Fallback to paragraphs in the entry content
        print("⚠️ No structured steps found, falling back to general content.")
        paras = soup.select("div.entry-content p")
        for p in paras:
            text = p.get_text(strip=True)
            if text and len(text) > 30 and not text.lower().startswith("watch the video"):  # Filter out short/meta lines
                steps.append(text)

    if not steps:
        print("❌ No recipe steps found.")
        return ["No cooking instructions found."]

    return steps
