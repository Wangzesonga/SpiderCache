import redis
from rediscluster import RedisCluster

class RedisWrapper:
    def __init__(self, redis_conn, prefix):
        self.redis_conn = redis_conn
        self.prefix = prefix
    
    def set(self, key, value):
        self.redis_conn.set(f'{self.prefix}{key}', value)
    
    def get(self, key):
        return self.redis_conn.get(f'{self.prefix}{key}')
    
    def delete(self, key):
        self.redis_conn.delete(f'{self.prefix}{key}')
    
    def exists(self, key):
        return self.redis_conn.exists(f'{self.prefix}{key}')
    
    def count_keys(self):
        count = 0
        cursor = 0
        while True:
            cursor, keys = self.redis_conn.scan(cursor=cursor, match=f'{self.prefix}*')
            count += len(keys)
            if cursor == 0:
                break
        return count

    def get_all_keys(self):
        keys = []
        cursor = 0
        while True:
            cursor, batch = self.redis_conn.scan(cursor=cursor, match=f'{self.prefix}*')
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def clear_all(self):
        keys = self.get_all_keys()
        if keys:
            self.redis_conn.delete(*keys) 
