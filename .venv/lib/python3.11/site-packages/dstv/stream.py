import time
import json
import os

from twisted.internet import defer

from zope.interface import implements

from tensor.interfaces import ITensorSource
from tensor.objects import Source
from tensor import utils

from dstv.lib import NexgenClient

import cv2

class DStvStream(Source):
    """
    """

    implements(ITensorSource)

    key = 'test:d48e83793c14d0f3f495f6066ba83b0031dbca69'
    url = 'http://controller.dstvmobile.com'

    def __init__(self, *a, **kw):
        Source.__init__(self, *a, **kw)

        self.client = NexgenClient(self.url, self.key)

    @defer.inlineCallbacks
    def get(self):

        channels = yield self.client.getChannels()

        for channel in channels:
            if channel['name'][:2] == 'RN':
                stream = yield self.getStream(channel['id'], 'vodacom-tanzania')

                out, err, code = yield self.getVideo(stream['stream_url', 'stream.mkv')

                if code == 0:
                    cap = cv2.VideoCapture('stream.mkv')

                    frames = 0
                    bright = []
                    while(cap.isOpened()):
                        ret, frame = cap.read()

                        if ret:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            r = cv2.calcHist([gray], [0], None, [256],[0,256])
                            cv2.normalize(r, r, 0, 255, cv2.NORM_MINMAX)

                            # Calculate the base intensity
                            v = sum(r)
                            ev = sum([i * (c/v) for i,c in enumerate(r)])[0]

                            frames += 1

                            bright.append(ev)
                        else:
                            break
                    cap.release()

                    if frames:
                        avsq = sum(bright)/frames
                    else:
                        avsq = 0

                    print avsq

                    defer.returnValue(
                        self.createEvent('ok', '%s SQ' % channel['name'],
                            avsq, prefix='channel.%s.sq' % channel['name'])
                    )
