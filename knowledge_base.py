"""Base de connaissances dermatologiques pour le RAG SkinDetect.

Contenu éducatif structuré en "chunks". Chaque condition possède plusieurs
entrées couvrant définition, symptômes, contagion, soins, prévention et
signes d'alarme. Le module rag.py récupère ces entrées par mots-clés.
"""

KNOWLEDGE_CHUNKS = [
    # ===================== VARICELLE (Chickenpox) =====================
    {
        "id": "chickenpox_1",
        "keywords": ["varicelle", "chickenpox", "vésicule", "bouton", "éruption", "gratter", "viral"],
        "content": (
            "La varicelle est une infection virale très contagieuse causée par le virus varicelle-zona (VZV). "
            "Elle provoque une éruption de petites vésicules remplies de liquide qui passent par stades : "
            "taches rouges, cloques, puis croûtes. L'éruption démange intensément et touche tout le corps "
            "ainsi que le cuir chevelu. Elle est très fréquente chez les enfants mais peut être grave chez "
            "l'adulte, la femme enceinte et les personnes immunodéprimées."
        ),
    },
    {
        "id": "chickenpox_2",
        "keywords": ["varicelle", "contagieux", "contagion", "isolement", "propagation", "transmission"],
        "content": (
            "La varicelle se transmet par contact direct avec les vésicules, la salive infectée et les "
            "gouttelettes respiratoires. Un patient est contagieux 1 à 2 jours avant l'éruption et jusqu'à "
            "ce que toutes les lésions soient croûtées (environ 5 à 7 jours). Il faut s'isoler des femmes "
            "enceintes, des nourrissons et des personnes immunodéprimées. Ne pas gratter les boutons pour "
            "éviter cicatrices et surinfections bactériennes."
        ),
    },
    {
        "id": "chickenpox_3",
        "keywords": ["varicelle", "soin", "compresse", "calmer", "démangeaison", "ongles", "fière"],
        "content": (
            "Soins de la varicelle : couper les ongles courts, porter des gants la nuit, appliquer des "
            "compresses fraîches et des crèmes apaisantes sur les lésions. Prendre des bains tièdes avec du "
            "bicarbonate ou de l'avoine colloïdale peut soulager les démangeaisons. Les vêtements amples en "
            "coton sont recommandés. Consulter un médecin en cas de forte fièvre, difficultés respiratoires, "
            "lésions très rouges/douleur (surinfection), ou chez une personne à risque."
        ),
    },

    # ===================== COWPOX (vaccine/cowpox) =====================
    {
        "id": "cowpox_1",
        "keywords": ["cowpox", "vaccine", "variole bovine", "animaux", "lésion", "zoonose", "rare"],
        "content": (
            "La vaccine (cowpox) est une infection virale rare de type zoonose, transmise par contact avec "
            "des animaux infectés (vaches, chats, rongeurs). Elle se manifeste par une lésion cutanée "
            "unique ou peu nombreuses, ulcérée, entourée de rougeur, souvent au niveau de la main. La "
            "lésion passe par une pustule puis une croûte et guérit généralement seule en quelques semaines."
        ),
    },
    {
        "id": "cowpox_2",
        "keywords": ["cowpox", "soin", "hygiène", "mains", "pansement", "infection", "lésion"],
        "content": (
            "En cas de cowpox : éviter de toucher la lésion et se laver soigneusement les mains. Couvrir la "
            "zone avec un pansement propre, ne pas partager les objets personnels (serviettes, vêtements) et "
            "surveiller les signes d'infection secondaire (pus, rougeur croissante, fièvre). Consulter un "
            "médecin pour confirmer le diagnostic et écarter d'autres infections virales plus graves."
        ),
    },

    # ===================== SYNDROME PIEDS-MAINS-BOUCHE =====================
    {
        "id": "hfmd_1",
        "keywords": ["pieds-mains-bouche", "hfmd", "enfant", "vésicule", "bouche", "fièvre", "entérovirus"],
        "content": (
            "Le syndrome pieds-mains-bouche (HFMD) est une infection virale courante chez les jeunes enfants, "
            "causée par des entérovirus (souvent Coxsackie A16). Il se manifeste par de la fièvre, des "
            "petites vésicules sur les paumes, la plante des pieds et dans la bouche, accompagnées de "
            "plaies buccales douloureuses. Il est très contagieux mais bénin dans la plupart des cas."
        ),
    },
    {
        "id": "hfmd_2",
        "keywords": ["pieds-mains-bouche", "soin", "hydratation", "déshydratation", "s'alimenter", "douleur"],
        "content": (
            "Soins du pieds-mains-bouche : l'hydratation est primordiale car les plaies buccales rendent la "
            "prise de boisson et de nourriture douloureuse. Proposer des aliments mous et froids, des "
            "boissons fraîches, éviter les aliments acides/salés. Se laver fréquemment les mains, désinfecter "
            "les surfaces et jouets, éviter le contact rapproché pendant 7 à 10 jours. Consulter si forte "
            "fièvre, déshydratation, maux de tête sévères ou raideur de la nuque."
        ),
    },

    # ===================== PEAU SAINE =====================
    {
        "id": "healthy_1",
        "keywords": ["sain", "healthy", "peau", "normal", "protection", "soleil", "hydratation"],
        "content": (
            "Une peau saine ne présente pas d'anomalie cutanée détectée. Pour la maintenir : protéger la "
            "peau du soleil avec un SPF 30 ou plus, maintenir une hydratation quotidienne, adopter une "
            "alimentation équilibrée riche en vitamines, et surveiller tout changement inhabituel "
            "(nouveaux grains de beauté, taches qui évoluent, lésions qui saignent). Un examen préventif "
            "annuel chez le dermatologue est recommandé, surtout si antécédents familiaux de cancer de la peau."
        ),
    },

    # ===================== ROUGEOLE =====================
    {
        "id": "measles_1",
        "keywords": ["rougeole", "measles", "éruption", "fièvre", "contagieux", "toux", "grippe"],
        "content": (
            "La rougeole est une infection virale très contagieuse et potentiellement grave, causée par le "
            "virus de la rougeole. Elle débute par de la fièvre, une toux, un écoulement nasal et une "
            "conjonctivite, suivis d'une éruption maculo-papuleuse qui part du visage et s'étend au corps. "
            "Les complications incluent pneumonie et encéphalite, surtout chez les jeunes enfants et les "
            "personnes immunodéprimées."
        ),
    },
    {
        "id": "measles_2",
        "keywords": ["rougeole", "isolement", "isolation", "complication", "urgence", "convulsion", "lumière"],
        "content": (
            "La rougeole impose une isolation stricte d'au moins 4 jours après l'apparition de l'éruption. "
            "Repos dans une pièce à lumière tamisée (photosensibilité), hydratation abondante, paracétamol "
            "pour la fièvre. Éviter tout contact avec les personnes non vaccinées, les femmes enceintes et "
            "les bébés. CONSULTATION MÉDICALE URGENTE en cas de difficultés respiratoires, convulsions ou "
            "confusion : la rougeole peut être mortelle."
        ),
    },

    # ===================== MONKEYPOX / MPOX =====================
    {
        "id": "monkeypox_1",
        "keywords": ["mpox", "monkeypox", "variole du singe", "lésion", "éruption", "vésicule", "virus"],
        "content": (
            "La variole du singe (Mpox) est une infection virale (genre Orthopoxvirus) avec éruption "
            "cutanée caractéristique : lésions qui évoluent de macules vers papules, vésicules, pustules "
            "puis croûtes. Elle s'accompagne souvent de fièvre, maux de tête, adénopathies (ganglions "
            "gonflés) et fatigue. La transmission se fait par contact étroit avec les lésions, les "
            "sécrétions respiratoires et les objets contaminés."
        ),
    },
    {
        "id": "monkeypox_2",
        "keywords": ["mpox", "isolement", "isolation", "pansement", "désinfecter", "linge", "déclaration"],
        "content": (
            "En cas de Mpox : ISOLEMENT REQUIS, éviter tout contact physique jusqu'à guérison complète. "
            "Couvrir les lésions avec des pansements et les changer régulièrement, désinfecter les surfaces "
            "touchées, laver le linge à haute température, et ne partager aucun objet personnel (literie, "
            "vêtements, ustensiles). La maladie nécessite un suivi médical strict ; signaler le cas aux "
            "autorités sanitaires selon les recommandations locales."
        ),
    },

    # ===================== CONSEILS GÉNÉRAUX =====================
    {
        "id": "general_1",
        "keywords": ["fièvre", "urgence", "consulter", "médecin", "grave", "danger", "alarme", "symptôme"],
        "content": (
            "Signes d'alarme nécessitant une consultation rapide ou urgente : forte fièvre (>39°C) "
            "persistante, difficultés respiratoires, confusion, convulsions, déshydratation, lésions qui "
            "s'infectent (pus, rougeur croissante, douleur), éruption étendue ou qui s'étend rapidement, et "
            "état général altéré. Ces signes concernent particulièrement les nourrissons, femmes enceintes, "
            "personnes âgées et immunodéprimées."
        ),
    },
    {
        "id": "general_2",
        "keywords": ["hygiène", "prévention", "mains", "désinfection", "propagation", "éviter", "quotidien"],
        "content": (
            "Mesures générales d'hygiène et de prévention des infections cutanées : se laver fréquemment les "
            "mains à l'eau et au savon, ne pas partager serviettes/vêtements/objets personnels, couvrir les "
            "lésions, désinfecter les surfaces et poignées, éternuer dans le coude, et éviter tout contact "
            "rapproché avec les personnes vulnérables tant que les lésions sont actives."
        ),
    },
    {
        "id": "general_3",
        "keywords": ["diagnostic", "ia", "modèle", "image", "analyse", "confiance", "interprétation"],
        "content": (
            "Les résultats fournis par SkinDetect sont générés par un modèle d'intelligence artificielle "
            "d'analyse d'image. Une prédiction avec une faible probabilité de confiance doit être "
            "interprétée avec précaution. Ceci ne constitue JAMAIS un diagnostic médical officiel : seul un "
            "professionnel de santé formé peut établir un diagnostic. En cas de doute, consultez un "
            "dermatologue ou un médecin traitant."
        ),
    },
    {
        "id": "general_4",
        "keywords": ["bébé", "nourrisson", "enceinte", "grossesse", "immunodéprimé", "vulnerable", "risque"],
        "content": (
            "Les personnes à risque de complications lors d'une infection cutanée virale sont les nourrissons "
            "de moins de 6 mois, les femmes enceintes (risque pour le fœtus ou le nouveau-né), les personnes "
            "âgées et les personnes immunodéprimées (chimiothérapie, VIH, corticoïdes au long cours, etc.). "
            "Un contact avec ces personnes impose une vigilance accrue et une consultation médicale rapide."
        ),
    },
]
