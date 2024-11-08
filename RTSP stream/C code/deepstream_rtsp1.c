#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

static GstRTSPServer *server = NULL;
static GstRTSPMediaFactory *factory = NULL;

static void on_pad_added(GstElement *src, GstPad *new_pad, GstElement *depay);

int main(int argc, char *argv[]) {
    GstElement *pipeline, *source, *depay, *decoder, *convert, *encoder, *rtppay;

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Initialize the RTSP server
    server = gst_rtsp_server_new();
    factory = gst_rtsp_media_factory_new();

    // Create elements for receiving RTSP input
    source = gst_element_factory_make("rtspsrc", "source");
    depay = gst_element_factory_make("rtph264depay", "depay");
    decoder = gst_element_factory_make("omxh264dec", "decoder");
    convert = gst_element_factory_make("videoconvert", "convert");
    encoder = gst_element_factory_make("x264enc", "encoder");
    rtppay = gst_element_factory_make("rtph264pay", "rtppay");

    if (!source || !depay || !decoder || !convert || !encoder || !rtppay) {
        g_printerr("One or more elements could not be created.\n");
        return -1;
    }

    // Set RTSP input source URI
    g_object_set(source, "location", "rtsp://192.168.0.103:8554/stream", "timeout", (guint64)30 * GST_SECOND, NULL);

    // Create the pipeline and add elements
    pipeline = gst_pipeline_new("rtsp-in-out-pipeline");
    gst_bin_add_many(GST_BIN(pipeline), source, depay, decoder, convert, encoder, rtppay, NULL);

    // Link elements for dynamic RTSP source
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), depay);

    // Link remaining static elements
    if (!gst_element_link_many(depay, decoder, convert, encoder, rtppay, NULL)) {
        g_printerr("Failed to link elements.\n");
        return -1;
    }

    // Set pipeline to play state
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Set up the RTSP media factory to output the processed stream
    gst_rtsp_media_factory_set_launch(factory, 
        "( rtspsrc location=rtsp://192.168.0.103:8554/stream ! rtph264depay ! avdec_h264 ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96 )");

    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/ds-test", factory);  // Change '/output' to '/ds-test'

    // Start the RTSP server
    g_print("RTSP server started at rtsp://192.168.0.231:8554/ds-test\n");
    gst_rtsp_server_attach(server, NULL);

    // Run the main loop
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);

    // Cleanup
    gst_object_unref(mounts);
    gst_object_unref(server);
    gst_object_unref(pipeline);

    return 0;
}

// Handler for the dynamic source pad connection
static void on_pad_added(GstElement *src, GstPad *new_pad, GstElement *depay) {
    GstPad *sink_pad = gst_element_get_static_pad(depay, "sink");

    g_print("Received new pad '%s' from '%s':\n", GST_PAD_NAME(new_pad), GST_ELEMENT_NAME(src));

    if (gst_pad_is_linked(sink_pad)) {
        g_print("Pad is already linked. Ignoring.\n");
        g_object_unref(sink_pad);
        return;
    }

    // Attempt to link the new pad with depay's sink pad
    GstPadLinkReturn ret = gst_pad_link(new_pad, sink_pad);
    if (ret != GST_PAD_LINK_OK) {
        g_printerr("Failed to link new pad to depay sink pad. Reason: %d\n", ret);
    } else {
        g_print("Successfully linked new pad to depay sink pad.\n");
    }

    g_object_unref(sink_pad);
}
