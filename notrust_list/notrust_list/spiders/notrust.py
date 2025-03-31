from typing import Iterable
import scrapy
import re
from notrust_list.items import NotrustListItem

class NotrustSpider(scrapy.Spider):
    name = "notrust"
    allowed_domains = ["zakupki.gov.ru"]
    start_urls = ["https://zakupki.gov.ru/epz/dishonestsupplier"]
    def start_requests(self):
        for i in range(1, 101):  #CHANGE
            url_start = f'https://zakupki.gov.ru/epz/dishonestsupplier/search/results.html?searchString=&morphology=on&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D0%BE%D0%B1%D0%BD%D0%BE%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F&savedSearchSettingsIdHidden=&sortBy=UPDATE_DATE&pageNumber={i}&sortDirection=false&recordsPerPage=_10&showLotsInfoHidden=false&fz94=on&fz223=on&ppRf615=on&dsStatuses=&inclusionDateFrom=&inclusionDateTo=&lastUpdateDateFrom=&lastUpdateDateTo='
            yield scrapy.Request(url=url_start, callback=self.parse)

    def parse(self, response):
        blocks = response.xpath('//div[@class="search-registry-entry-block box-shadow-search-input"]')
        for block in blocks:
            suplier = NotrustListItem() 
            suplier['suplier_name'] = block.xpath('.//*[@class="registry-entry__body"]/div[@class="registry-entry__body-block"][1]//*[@class="registry-entry__body-value"]/text()').extract()[0]
            suplier['suplier_inn'] = block.xpath('.//*[@class="registry-entry__body"]/div[@class="registry-entry__body-block"][2]//*[@class="registry-entry__body-value"]/text()').extract()[0]
            status = block.xpath('.//*[@class="registry-entry__header-mid__title"]/text()').extract()
            if status == "Размещено":
                suplier['no_trust_now'] = 1
                suplier['no_trust_before'] = 0
            else:
                suplier['no_trust_now'] = 0
                suplier['no_trust_before'] = 1
            yield suplier


#scrapy crawl notrust -o res.json
            