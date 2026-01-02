#!/usr/bin/env python3
"""
Generate training data for query intent classification.
Creates labeled examples of questions vs keyword searches in multiple languages.
Includes real queries from production search logs.
"""

import json
import random
import re

# Question templates - natural language queries
QUESTION_TEMPLATES = {
    "en": [
        "What is {topic}?",
        "How does {topic} work?",
        "Why is {topic} important?",
        "Who invented {topic}?",
        "Where can I learn about {topic}?",
        "When did {topic} start?",
        "Which {topic} is best?",
        "Can you explain {topic}?",
        "What are the benefits of {topic}?",
        "How do I get started with {topic}?",
        "What's the difference between {topic} and {topic2}?",
        "Is {topic} worth learning?",
        "Tell me about {topic}",
        "I want to understand {topic}",
        "Explain {topic} to me",
        "What are the latest trends in {topic}?",
        "How will {topic} change in the future?",
        "What do experts say about {topic}?",
        "Are there any risks with {topic}?",
        "What should I know about {topic}?",
        "Can {topic} help with {topic2}?",
        "Does {topic} really work?",
        "Is there a connection between {topic} and {topic2}?",
    ],
    "de": [
        "Was ist {topic}?",
        "Wie funktioniert {topic}?",
        "Warum ist {topic} wichtig?",
        "Wer hat {topic} erfunden?",
        "Wo kann ich {topic} lernen?",
        "Wann hat {topic} begonnen?",
        "Welches {topic} ist am besten?",
        "Kannst du {topic} erklären?",
        "Was sind die Vorteile von {topic}?",
        "Wie fange ich mit {topic} an?",
        "Erzähl mir von {topic}",
        "Ich möchte {topic} verstehen",
    ],
    "es": [
        "¿Qué es {topic}?",
        "¿Cómo funciona {topic}?",
        "¿Por qué es importante {topic}?",
        "¿Quién inventó {topic}?",
        "¿Dónde puedo aprender {topic}?",
        "¿Cuándo comenzó {topic}?",
        "¿Cuál {topic} es mejor?",
        "¿Puedes explicar {topic}?",
        "¿Cuáles son los beneficios de {topic}?",
        "¿Cómo empiezo con {topic}?",
        "Cuéntame sobre {topic}",
        "Quiero entender {topic}",
    ],
    "fr": [
        "Qu'est-ce que {topic}?",
        "Comment fonctionne {topic}?",
        "Pourquoi {topic} est important?",
        "Qui a inventé {topic}?",
        "Où puis-je apprendre {topic}?",
        "Quand {topic} a-t-il commencé?",
        "Quel {topic} est le meilleur?",
        "Peux-tu expliquer {topic}?",
        "Quels sont les avantages de {topic}?",
        "Comment commencer avec {topic}?",
        "Parle-moi de {topic}",
        "Je veux comprendre {topic}",
    ],
    "pt": [
        "O que é {topic}?",
        "Como funciona {topic}?",
        "Por que {topic} é importante?",
        "Quem inventou {topic}?",
        "Onde posso aprender {topic}?",
        "Quando {topic} começou?",
        "Qual {topic} é melhor?",
        "Pode explicar {topic}?",
        "Quais são os benefícios de {topic}?",
        "Como começar com {topic}?",
        "Me conte sobre {topic}",
        "Quero entender {topic}",
    ],
    "it": [
        "Cos'è {topic}?",
        "Come funziona {topic}?",
        "Perché {topic} è importante?",
        "Chi ha inventato {topic}?",
        "Dove posso imparare {topic}?",
        "Quando è iniziato {topic}?",
        "Quale {topic} è migliore?",
        "Puoi spiegare {topic}?",
        "Quali sono i vantaggi di {topic}?",
        "Come iniziare con {topic}?",
        "Parlami di {topic}",
        "Voglio capire {topic}",
    ],
    "nl": [
        "Wat is {topic}?",
        "Hoe werkt {topic}?",
        "Waarom is {topic} belangrijk?",
        "Wie heeft {topic} uitgevonden?",
        "Waar kan ik {topic} leren?",
        "Wanneer begon {topic}?",
        "Welke {topic} is het beste?",
        "Kun je {topic} uitleggen?",
        "Wat zijn de voordelen van {topic}?",
        "Hoe begin ik met {topic}?",
        "Vertel me over {topic}",
    ],
    "pl": [
        "Co to jest {topic}?",
        "Jak działa {topic}?",
        "Dlaczego {topic} jest ważny?",
        "Kto wynalazł {topic}?",
        "Gdzie mogę nauczyć się {topic}?",
        "Kiedy {topic} się zaczął?",
        "Który {topic} jest najlepszy?",
        "Czy możesz wyjaśnić {topic}?",
        "Jakie są korzyści z {topic}?",
        "Jak zacząć z {topic}?",
        "Opowiedz mi o {topic}",
    ],
    "ru": [
        "Что такое {topic}?",
        "Как работает {topic}?",
        "Почему {topic} важен?",
        "Кто изобрёл {topic}?",
        "Где я могу изучить {topic}?",
        "Когда началось {topic}?",
        "Какой {topic} лучше?",
        "Можешь объяснить {topic}?",
        "Каковы преимущества {topic}?",
        "Как начать с {topic}?",
        "Расскажи мне о {topic}",
    ],
    "zh": [
        "{topic}是什么?",
        "{topic}如何工作?",
        "为什么{topic}很重要?",
        "谁发明了{topic}?",
        "我在哪里可以学习{topic}?",
        "{topic}什么时候开始的?",
        "哪个{topic}最好?",
        "你能解释{topic}吗?",
        "{topic}有什么好处?",
        "如何开始学习{topic}?",
        "告诉我关于{topic}的事情",
    ],
    "ja": [
        "{topic}とは何ですか?",
        "{topic}はどのように機能しますか?",
        "なぜ{topic}は重要ですか?",
        "誰が{topic}を発明しましたか?",
        "{topic}をどこで学べますか?",
        "{topic}はいつ始まりましたか?",
        "どの{topic}が一番いいですか?",
        "{topic}を説明してくれますか?",
        "{topic}の利点は何ですか?",
        "{topic}の始め方は?",
        "{topic}について教えてください",
    ],
    "ko": [
        "{topic}이란 무엇인가요?",
        "{topic}은 어떻게 작동하나요?",
        "왜 {topic}이 중요한가요?",
        "누가 {topic}을 발명했나요?",
        "{topic}을 어디서 배울 수 있나요?",
        "{topic}은 언제 시작되었나요?",
        "어떤 {topic}이 가장 좋나요?",
        "{topic}을 설명해 줄 수 있나요?",
        "{topic}의 장점은 무엇인가요?",
        "{topic}을 시작하는 방법은?",
        "{topic}에 대해 알려주세요",
    ],
}

# Keyword patterns - short, specific searches
KEYWORD_PATTERNS = [
    "{topic}",
    "{topic} {topic2}",
    "{topic} podcast",
    "{topic} interview",
    "{topic} episode",
    "{topic} 2024",
    "{topic} 2025",
    "{topic} 2026",
    "{topic} news",
    "{topic} latest",
    "{topic} best",
    "{topic} top",
    "{topic} review",
    "{person}",
    "{person} {topic}",
    "{person} interview",
    "{person} podcast",
    "{person} on {topic}",
    "{person} talks about {topic}",
    "\"{topic}\"",  # Quoted search
    "{topic} vs {topic2}",
    "{topic} AND {topic2}",
    "{topic} OR {topic2}",
    "{topic} -spam",
    "best {topic} podcasts",
    "top {topic} episodes",
    "{topic} healthcare",
    "{topic} diagnosis",
    "{topic} future predictions",
    "\"{person}\"",
    "{topic} NOT {topic2}",
    "\"{topic}\" AND \"{topic2}\"",
    # Short compound nouns - explicitly keywords
    "{compound}",
]

# Compound noun phrases that are clearly keywords (not questions)
COMPOUND_KEYWORDS = [
    "bitcoin mining", "machine learning", "deep learning", "neural network",
    "data science", "computer vision", "natural language", "speech recognition",
    "self driving", "autonomous vehicles", "electric cars", "solar power",
    "wind energy", "nuclear power", "quantum computing", "edge computing",
    "cloud storage", "cyber security", "data privacy", "user experience",
    "product design", "growth hacking", "content marketing", "email marketing",
    "social media", "influencer marketing", "brand strategy", "market research",
    "supply chain", "real estate", "private equity", "hedge fund",
    "stock market", "bond market", "forex trading", "options trading",
    "venture capital", "angel investing", "seed funding", "series A",
    "artificial intelligence", "augmented reality", "virtual reality", "mixed reality",
    "internet things", "smart home", "wearable technology", "fitness tracker",
    "mental health", "physical therapy", "weight loss", "muscle building",
    "healthy eating", "intermittent fasting", "keto diet", "vegan lifestyle",
    "remote work", "digital nomad", "work life balance", "career development",
    "personal branding", "public speaking", "leadership skills", "team management",
    "project management", "agile methodology", "scrum master", "product owner",
    "software engineering", "frontend development", "backend development", "full stack",
    "mobile development", "iOS development", "Android development", "cross platform",
    "web development", "responsive design", "user interface", "user research",
    "A/B testing", "conversion optimization", "search engine", "paid advertising",
    "podcast production", "video editing", "audio engineering", "music production",
    "true crime", "comedy podcast", "news podcast", "interview podcast",
    "tech news", "business news", "sports news", "entertainment news",
]

# Topics for substitution
TOPICS = [
    "artificial intelligence", "machine learning", "deep learning", "neural networks",
    "cryptocurrency", "bitcoin", "blockchain", "NFT",
    "climate change", "renewable energy", "sustainability", "electric vehicles",
    "startup", "entrepreneurship", "venture capital", "fundraising",
    "productivity", "time management", "remote work", "leadership",
    "health", "fitness", "nutrition", "mental health", "meditation",
    "investing", "stock market", "real estate", "personal finance",
    "programming", "software development", "web development", "mobile apps",
    "marketing", "social media", "content creation", "SEO",
    "science", "physics", "biology", "chemistry", "astronomy",
    "history", "politics", "economics", "philosophy",
    "AI", "ML", "GPT", "LLM", "AGI",
    "podcast", "audio", "music", "technology",
    "business", "finance", "economy", "market",
    "healthcare", "diagnosis", "radiology", "value based care",
    "workflow automation", "ai agents", "decision intelligence",
]

TOPICS_SHORT = [
    "AI", "ML", "crypto", "bitcoin", "startup", "tech", "health",
    "fitness", "investing", "coding", "marketing", "science",
    "grok", "OpenAI", "Claude", "GPT", "trump", "musk",
]

# People for substitution
PEOPLE = [
    "Elon Musk", "Sam Altman", "Joe Rogan", "Tim Ferriss", "Lex Fridman",
    "Naval Ravikant", "Marc Andreessen", "Peter Thiel", "Reid Hoffman",
    "Satya Nadella", "Sundar Pichai", "Jensen Huang", "Mark Zuckerberg",
    "Andrew Huberman", "Jordan Peterson", "Ben Shapiro", "Russell Brand",
    "Gary Vaynerchuk", "Tony Robbins", "Brené Brown", "Simon Sinek",
    "Cassie Kozyrkov", "Emily Bender", "Jerome Powell", "Donald Trump",
]

def is_question_heuristic(query: str) -> bool:
    """Use heuristics to guess if a query is a question."""
    q = query.lower().strip()

    # Ends with question mark (but not just "Name?")
    if q.endswith('?') and len(q.split()) > 2:
        return True

    # Starts with question words
    question_starters = [
        "what ", "how ", "why ", "who ", "where ", "when ", "which ",
        "is ", "are ", "can ", "do ", "does ", "tell me", "explain",
        "was ", "wie ", "warum ", "wer ", "wo ", "wann ",  # German
        "qué ", "cómo ", "por qué ", "quién ",  # Spanish
        "qu'est", "comment ", "pourquoi ",  # French
        "o que ", "como ", "por que ",  # Portuguese
    ]
    for starter in question_starters:
        if q.startswith(starter):
            return True

    # Contains phrases that indicate questions
    if "can you" in q or "could you" in q or "i want to know" in q:
        return True

    return False

def load_real_queries():
    """Load real queries from production and label them."""
    try:
        with open("real_queries.txt", "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: real_queries.txt not found, using only synthetic data")
        return []

    labeled = []
    for query in queries:
        if not query or len(query) < 3:
            continue

        label = "question" if is_question_heuristic(query) else "keyword"
        labeled.append({"query": query, "label": label, "lang": "real", "source": "production"})

    return labeled

def generate_questions(n=500):
    """Generate n question examples."""
    examples = []

    for _ in range(n):
        lang = random.choice(list(QUESTION_TEMPLATES.keys()))
        template = random.choice(QUESTION_TEMPLATES[lang])
        topic = random.choice(TOPICS)
        topic2 = random.choice(TOPICS)

        query = template.format(topic=topic, topic2=topic2)
        examples.append({"query": query, "label": "question", "lang": lang, "source": "synthetic"})

    return examples

def generate_keywords(n=500):
    """Generate n keyword search examples."""
    examples = []

    for _ in range(n):
        pattern = random.choice(KEYWORD_PATTERNS)
        topic = random.choice(TOPICS + TOPICS_SHORT)
        topic2 = random.choice(TOPICS + TOPICS_SHORT)
        person = random.choice(PEOPLE)
        compound = random.choice(COMPOUND_KEYWORDS)

        query = pattern.format(topic=topic, topic2=topic2, person=person, compound=compound)
        examples.append({"query": query, "label": "keyword", "lang": "en", "source": "synthetic"})

    # Add all compound keywords explicitly
    for compound in COMPOUND_KEYWORDS:
        examples.append({"query": compound, "label": "keyword", "lang": "en", "source": "synthetic"})

    return examples

def main():
    print("Generating training data...")

    # Synthetic data
    questions = generate_questions(500)
    keywords = generate_keywords(500)

    # Real production data
    real_data = load_real_queries()
    print(f"Loaded {len(real_data)} real queries from production")

    # Count real questions vs keywords
    real_questions = sum(1 for x in real_data if x["label"] == "question")
    real_keywords = sum(1 for x in real_data if x["label"] == "keyword")
    print(f"Real data: {real_questions} questions, {real_keywords} keywords")

    # Combine all data
    all_data = questions + keywords + real_data
    random.shuffle(all_data)

    # Split into train/test
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # Save
    with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(train_data)} training examples")
    print(f"Generated {len(test_data)} test examples")

    # Stats
    train_questions = sum(1 for x in train_data if x["label"] == "question")
    train_keywords = sum(1 for x in train_data if x["label"] == "keyword")
    print(f"Training: {train_questions} questions, {train_keywords} keywords")

    # Source breakdown
    train_synthetic = sum(1 for x in train_data if x.get("source") == "synthetic")
    train_real = sum(1 for x in train_data if x.get("source") == "production")
    print(f"Training sources: {train_synthetic} synthetic, {train_real} real")

    # Sample
    print("\nSample questions:")
    for ex in random.sample([x for x in train_data if x["label"] == "question"], 5):
        src = ex.get("source", "?")[:4]
        print(f"  [{src}] {ex['query'][:70]}")

    print("\nSample keywords:")
    for ex in random.sample([x for x in train_data if x["label"] == "keyword"], 5):
        src = ex.get("source", "?")[:4]
        print(f"  [{src}] {ex['query'][:70]}")

if __name__ == "__main__":
    main()
