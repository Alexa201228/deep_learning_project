import asyncio

import httpx as httpx
import pandas as pd
from loguru import logger
from bs4 import BeautifulSoup


async def get_data_to_dataframe() -> dict:
    """
    Function to get data from https://scitechdaily.com
    and load it to dict
    :return:
    """
    articles_data = {"category_tag": [], "article_title": [], "article_link": [], "article_content": []}

    # For the first page we should get content separately because of different element tag that contains content
    await extract_data(articles_data, 1, "class", "archive-list")
    for page_num in range(2, 101):
        await extract_data(articles_data, page_num, "id", "main-content")

    return articles_data


async def extract_data(data_dict: dict, page_num: int, main_content_attribute: str, main_content_tag: str):
    """
    Method to parse data from HTML page
    :param data_dict: dict to dump data
    :param page_num: number of page to download
    :param main_content_attribute: div attribute of main content
    :param main_content_tag: definition of attribute (e.g. class name)
    :return:
    """
    client = httpx.AsyncClient(verify=False, follow_redirects=True)
    content = await client.get(f"https://scitechdaily.com/page/{page_num}")
    bs = BeautifulSoup(content.text)
    articles = bs.find("div", {main_content_attribute: main_content_tag}).find_all("article",
                                                                                   class_=["content-list", "clearfix"])
    for article in articles:
        category_tag = article.find_all("span", {"class": "entry-meta-cats"})[0]
        article_title = article.find_all("h3", class_=["entry-title", "content-list-title"])[0]
        data_dict["category_tag"].append(category_tag.a.text)
        data_dict["article_title"].append(article_title.a["title"])
        data_dict["article_link"].append(article_title.a["href"])
        article_content = await client.get(article_title.a["href"])
        article_bs = BeautifulSoup(article_content.text)
        article_text = article_bs.find("div", {"class": "entry-content"}).find_all("p")
        raw_content = [p.text for p in article_text]
        data_dict["article_content"].append("\n".join(raw_content))

    logger.info(f"Data from page {page_num=} is downloaded")


def dump_data_to_csv(data: dict) -> None:
    """
    Method to dump data to csv file in train folder
    :param data: dict with keys as column names and values as lists
    :return:
    """

    df = pd.DataFrame(data)
    df.to_csv("./../data/train/scitechdaily.csv", index=False)


async def main():
    data: dict = await get_data_to_dataframe()
    dump_data_to_csv(data)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    logger.info("Download completed")
