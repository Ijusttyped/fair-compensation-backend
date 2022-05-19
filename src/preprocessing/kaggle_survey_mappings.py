""" Module containing all manual mappings for data harmonization """
from typing import Dict

GENDER_MAPPING = {"M": "male", "F": "female", "D": "diverse"}

CITY_MAPPING = {
    "München": "Munich",
    "Köln": "Cologne",
    "Nürnberg": "Nuremberg",
    "Düsseldorf": "Dusseldorf",
    "Kiev": "Kyiv",
    "Saint-Petersburg": "Saint Petersburg",
}

POSITION_MAPPING = {
    "QA": "QA Engineer",
    "ML Engineer": "Machine Learning Engineer",
    "Backend Engineer": "Backend Developer",
    "Frontend Engineer": "Frontend Developer",
    "Front End Developer": "Frontend Developer",
    "Python Dev": "Software Developer",
    "Devops": "Devops Engineer",
    "Lead Devops": "Devops Engineer",
    "Full Stack Developer": "Fullstack Developer",
    "Full-Stack Developer": "Fullstack Developer",
    "Fullstack": "Fullstack Developer",
    "Developer": "Software Developer",
    "Big Data Engineer": "Data Engineer",
    "Lead Software Engineer": "Software Engineer",
    "Principal Software Engineer": "Software Engineer",
    "Web Developer": "Frontend Developer",
    "iOS Developer": "Mobile Developer",
    "Android Developer": "Mobile Developer",
    "SRE": "Site Reliability Engineer",
    "CTO (CEO, CFO)": "CTO",
    "Designer (UI/UX)": "UI, UX Designer",
    "Designer (UI, UX)": "UI, UX Designer",
    "UI/UX Designer": "UI, UX Designer",
    "UX Designer": "UI, UX Designer",
    "Head of Engineering": "Manager",
    "PM": "Manager",
}

POSITION_REGEX_MAPPING = {
    r"^((?i)(?:Java|Python|PHP|.net|C\+\+|Scala|Ruby|Javascript|c#|Oracle|Go|Golang)"
    + r"\s*(?:Software)?\s*(?:Developer))$": "Software Developer",
    r"^((?i)(?:Product|Project|QA|IT|Engineering|Program|Devops|Operations|Team|Technical Product|Technical Project)"
    + r"\s*(?:Manager))$": "Manager",
    r"^((?i)(?:SAP|BI|SAP BW|IT)\s*(?:Consultant))$": "Consultant",
    r"^((?i)(?:Solution|Solutions|Cloud|IT|QA TA|System)\s*(?:Architect))$": "Architect",
}

SENIORITY_MAPPING = {
    "Working Student": "Student",
    "Intern": "Student",
    "Entry Level": "Junior",
}

COMPANY_TYPE_MAPPING = {
    "Consulting / Agency": "Consulting or Agency",
    "Agency": "Consulting or Agency",
    "Consulting": "Consulting or Agency",
    "ECommerce": "E-Commerce",
    "Bank": "Finance",
    "Fintech": "Finance",
    "Financial": "Finance",
    "University": "Research and Education",
    "Research": "Research and Education",
    "Education": "Research and Education",
    "Research Institute": "Research and Education",
    "Institute": "Research and Education",
    "IT-Outsourcing": "Outsource",
    "Outsource": "Outsource",
    "Bodyshop / Outsource": "Outsource",
    "Outsourse": "Outsource",
    "Outsorce": "Outsource",
}

COMPANY_SIZE_MAPPING = {
    "10-50": "1-100",
    "up to 10": "1-100",
    "11-50": "1-100",
    "51-100": "1-100",
    "50-100": "1-100",
    "100-1000": "101-1000",
}


def lower_key_and_values(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Lowers the key and the value elements of the dictionary.
    Args:
        mapping (Dict[str, str]: The dictionary to apply.

    Returns:
        Dictionary with lowered elements.
    """
    return {k.lower(): v.lower() for k, v in mapping.items()}


MAPPINGS = {
    "Gender": lower_key_and_values(GENDER_MAPPING),
    "City": lower_key_and_values(CITY_MAPPING),
    "Position": lower_key_and_values(POSITION_MAPPING),
    "Seniority": lower_key_and_values(SENIORITY_MAPPING),
    "Company_Type": lower_key_and_values(COMPANY_TYPE_MAPPING),
    "Company_Size": lower_key_and_values(COMPANY_SIZE_MAPPING),
}


REGEX_MAPPINGS = {
    "Position": lower_key_and_values(POSITION_REGEX_MAPPING),
}
