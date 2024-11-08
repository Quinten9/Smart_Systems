import argparse
import sys
import gi
import platform
import math
import time
from gi.repository import GObject, Gst, GstRtspServer

MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"

def bus_call(bus, message, loop):
    """Handles messages from the GStreamer bus."""
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error:", err, debug)
        loop.quit()
    return True


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick Nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None
    return nbin


def main(args):
    number_sources = len(args)

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()
    is_live = False

    # Create streammux for multi-stream handling
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    pipeline.add(streammux)
    for i in range(number_sources):
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)
        sinkpad = streammux.get_request_pad("sink_%u" % i)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # Create video processing elements
    print("Creating tiler\n")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    print("Creating nvvidconv\n")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    print("Creating nvvidconv_postosd\n")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")

    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    encoder.set_property("bitrate", 4000000)  # You can adjust bitrate here

    if platform.machine() == "aarch64":
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        encoder.set_property("bufapi-version", 1)

    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", 5400)
    sink.set_property("async", False)
    sink.set_property("sync", 1)

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)

    tiler.set_property("rows", int(math.sqrt(number_sources)))
    tiler.set_property("columns", int(math.ceil((1.0 * number_sources) / int(math.sqrt(number_sources))))),
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    # Add elements to the pipeline
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Link elements
    streammux.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # Set up bus and event handling
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # RTSP setup
    rtsp_port_num = 8554
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch('( udpsrc name=pay0 port=5400 buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96 " )')
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print("\n *** RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except BaseException:
        pass

    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help')
    parser.add_argument("-i", "--input", help="Path to input stream", nargs="+", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.input)
