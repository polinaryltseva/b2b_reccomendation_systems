import scrapy
from zakgov.items import ZakgovItem
import pandas as pd
import polars as pl


def get_data():
    df = pd.read_excel('/Users/renatavaliullina/Desktop/rec b2b/zakgov/data_for_parsing_2020.xlsx', dtype={"pub_num": str})
    df = df[df['pub_num'].str.len()!=7]
    return df #[13993:14000]#[13882:1390]



class OzoikSpider(scrapy.Spider):
    name = "OzOik"
    allowed_domains = ["zakupki.gov.ru"]
    start_urls = ["https://zakupki.gov.ru/epz/order/extendedsearch"]
    custom_settings = {
        'DUPEFILTER_DEBUG': True,
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter'
    } 

    def start_requests(self):
        data = get_data() #data.shape[0]
        for i in range(data.shape[0]): 
            tender = data.iloc[i].to_list() # Keeps original types
            pub_num = tender[1]
            if len(pub_num) == 19:
                url = 'https://zakupki.gov.ru/epz/order/notice/ea44/view/common-info.html?regNumber='+pub_num
                yield scrapy.Request(url=url, callback=self.parse_tender, meta={'pub_num':pub_num, 'len':19})
            if len(pub_num) == 11:
                url = 'https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=' + pub_num + '&morphology=on&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F&pageNumber=1&sortDirection=false&recordsPerPage=_10&showLotsInfoHidden=false&sortBy=UPDATE_DATE&fz223=on&currencyIdGeneral=-1'
                yield scrapy.Request(url=url, callback=self.parse_search, meta={'pub_num':pub_num, 'len':11})
    
    def parse_search(self, response):
        search_res = response.xpath('//div[@class="search-registry-entry-block box-shadow-search-input"]')
        url = search_res[0].xpath('//div[@class="registry-entry__header-mid__number"]/a[@target="_blank"]/@href').extract()[0]
        url = response.urljoin(url)
        yield scrapy.Request(url=url, callback=self.parse_tender, meta=response.meta)



    def parse_tender(self, response):
        info = ZakgovItem()
        info['pub_num'] = str(response.meta.get('pub_num'))
        info['price'] = None
        info['oz'] = None
        info['oik'] = None
        info['no_info'] = 0

        page_title = response.xpath('//head/title/text()').extract()[0]
        if 'Страница не найдена' in page_title:
            info['no_info'] = 1
            yield info

        if int(response.meta.get('len')) == 19:
            info['price'] = response.xpath('//div[@class="cardHeaderBlock"]//span[@class="cardMainInfo__content cost"]/text()').extract()[0]
        else:
            info['price'] = response.xpath('//div[@class="cardHeaderBlock"]//div[@class="price-block__value"]/text()').extract()[0]


        
        blocks = response.xpath('//section[@class="blockInfo__section section"]')
        for item in blocks:
            title = item.xpath('.//span[@class="section__title"]/text()').extract()
            if 'Размер обеспечения заявки' in title:
                info['oz'] = item.xpath('.//span[@class="section__info"]/text()').extract()[0]
            if 'Размер обеспечения исполнения контракта' in title:
                info['oik'] = item.xpath('.//span[@class="section__info"]/text()').extract()[0]
        yield info
        

#scrapy crawl OzOik


''' 
response.xpath('//div[@class="cardHeaderBlock"]//span[@class="cardMainInfo__content cost"]/text()').extract()
response.xpath('//div[@class="cardHeaderBlock"]//div[@class="price-block__value"]/text()').extract()
'''


'''
response.xpath('//section[@class="blockInfo__section section"]').extract()
response.xpath('//section[@class="blockInfo__section section"]')[-4].xpath('.//span[@class="section__title"]/text()').extract()
response.xpath('//section[@class="blockInfo__section section"]')[-4].xpath('.//span[@class="section__info"]/text()').extract()[0]
'''
        

