#myFirstScraper

from bs4 import BeautifulSoup
from urllib2 import urlopen

BASE_URL = "http://studitu.tubmlr.com"

def getHeadline(section_url):
    html = urlopen(section_url).read()
    soup = BeautifulSoup(html, "lxml")
    athing = soup.find("tr", "athing")
    #category_links = [BASE_URL + dd.a["href"] for dd in athing.findAll("dd")]
    topTitle = [BASE_URL + td.a["title"] for td in athing.findAll("td") ]
    return topTitle


print getHeadline(BASE_URL)