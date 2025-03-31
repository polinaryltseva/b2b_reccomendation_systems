# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NotrustListItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    suplier_name = scrapy.Field()
    suplier_inn = scrapy.Field()
    no_trust_now = scrapy.Field()
    no_trust_before = scrapy.Field()
