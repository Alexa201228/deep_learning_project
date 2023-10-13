from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import services

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("main_page.html", {"request": request})


@app.post("/translate")
async def translate_user_text(request: Request, uploaded_file: UploadFile = File(...)):
    """
    Translate user text
    :param request: User request
    :param uploaded_file: file with test to translate
    :return:
    """
    translation = await services.get_text_translation(uploaded_file.file.read())
    return templates.TemplateResponse("translation.html", {"request": request, "translation": translation})


@app.post("/get-recommendation")
async def get_recommendation(request: Request, uploaded_file: UploadFile = File(...)):
    """
    Get recommended articles for user
    :param request: User request
    :param uploaded_file: file with text on which model will predict article category
    :return: TemplateResponse
    """
    recommendations, theme = await services.get_recommendations_from_content(uploaded_file.file.read())
    return templates.TemplateResponse("recommendations.html", {"request": request, "recommendations": recommendations,
                                                               "theme": theme})