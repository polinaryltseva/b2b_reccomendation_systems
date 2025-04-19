# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ZakgovItem(scrapy.Item):
    pub_num = scrapy.Field()
    price = scrapy.Field()
    oz = scrapy.Field()
    oik = scrapy.Field()
    no_info = scrapy.Field()