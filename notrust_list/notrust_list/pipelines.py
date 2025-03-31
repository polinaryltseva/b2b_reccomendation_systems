# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import json
from itemadapter import ItemAdapter


class NotrustListPipeline:
    def open_spider(self, spider): 
        self.file = open('res.json', 'w')
        self.file.write('[\n')
        self.first_item = True

    def close_spider(self, spider): 
        self.file.write('\n]')
        self.file.close()

    def process_item(self, item, spider): 
        line = json.dumps(ItemAdapter(item).asdict(), ensure_ascii=False, indent=2, separators=(',', ': ') ) + ",\n"
        self.file.write(line)
        self.first_item = False
        return item



