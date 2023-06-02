import os
import time
import json
import hashlib

from twisted.internet import defer
from tensor import utils


class NexgenClient(object):
    def __init__(self, url, key):
        self.channelCache = None
        self.streamCache = {}
        self.url = url
        self.key = key

    @defer.inlineCallbacks
    def getStream(self, id, group):
        now = time.time()
        if id in self.streamCache:
            if self.streamCache[id][0] > (now - 5*60):
                print "Prefetched", id
                defer.returnValue(self.streamCache[id])

        data = {
            #"user_agent": "Mozilla/5.0 (Linux; U; Android 2.2; en-ca; GT-P1000M Build/FROYO) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "user_agent": "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_0 like Mac OS X; en-us) AppleWebKit/532.9 (KHTML, like Gecko) Version/4.0.5 Mobile/8A293 Safari/6531.22.7",
            "content_type": "channel",
            "content_id": id,
            "publish_group": group
        }

        stream = yield utils.HTTPRequest().getJson(self.url + '/api/v1/stream/', 'POST',
            headers={
                'Authorization': ['ApiKey %s' % self.key]
            },
            data=json.dumps(data)
        )

        self.streamCache[id] = (time.time(), stream)

        defer.returnValue(stream)

    @defer.inlineCallbacks
    def getChannels(self):
        if not self.channelCache:
            self.channelCache = yield utils.HTTPRequest().getJson(self.url + '/api/v1/channel/?limit=100',
                headers={'Authorization': ['ApiKey %s' % self.key]})

        defer.returnValue(self.channelCache['objects'])
       

    @defer.inlineCallbacks
    def getVideo(self, url, out, time=10):
        try:
            if os.path.exists(out):
                os.unlink(out)

            out, err, code = yield utils.fork('/usr/bin/avconv',
                args=(
                    '-i', url,
                    '-t', str(time),
                    '-c', 'copy', out
                ), timeout=60.0)
        except Exception, e:
            out, err, code = "", str(e), 1

        defer.returnValue((out, err, code))
