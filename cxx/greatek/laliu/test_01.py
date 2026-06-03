import argparse
import sys
import urllib.parse

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import GLib, Gst  # noqa: E402


DEFAULT_RTSP_URL = "rtsp://admin:jiankong123@192.168.23.19:554/cam/realmonitor?channel=1&subtype=0"


def _set_property_if_exists(obj: object, name: str, value: object) -> None:
    find_prop = getattr(obj, "find_property", None)
    if callable(find_prop) and find_prop(name) is not None:
        try:
            obj.set_property(name, value)
        except Exception:
            pass


def _extract_user_pass(rtsp_url: str) -> tuple[str | None, str | None]:
    parsed = urllib.parse.urlsplit(rtsp_url)
    if not parsed.username and not parsed.password:
        return None, None
    return parsed.username, parsed.password


def build_pipeline(rtsp_url: str, transport: str, latency_ms: int, user_agent: str) -> Gst.Pipeline:
    pipeline = Gst.Pipeline.new("rtsp-player")
    if pipeline is None:
        raise RuntimeError("创建 GStreamer pipeline 失败")

    src = Gst.ElementFactory.make("rtspsrc", "source")
    if src is None:
        raise RuntimeError("缺少 rtspsrc 插件（gst-plugins-good）")

    src.set_property("location", rtsp_url)

    transport_map = {
        "auto": None,
        "udp": 0x1,
        "udp-mcast": 0x2,
        "tcp": 0x4,
    }
    protocols_value = transport_map.get(transport, None)
    if protocols_value is not None:
        _set_property_if_exists(src, "protocols", protocols_value)

    _set_property_if_exists(src, "latency", int(latency_ms))
    _set_property_if_exists(src, "tcp-timeout", int(30_000_000))
    _set_property_if_exists(src, "timeout", int(30_000_000))
    _set_property_if_exists(src, "do-rtsp-keep-alive", True)
    _set_property_if_exists(src, "short-header", True)
    if user_agent:
        _set_property_if_exists(src, "user-agent", user_agent)

    username, password = _extract_user_pass(rtsp_url)
    if username is not None:
        _set_property_if_exists(src, "user-id", username)
    if password is not None:
        _set_property_if_exists(src, "user-pw", password)

    decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
    if decodebin is None:
        raise RuntimeError("缺少 decodebin")

    videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
    if videoconvert is None:
        raise RuntimeError("缺少 videoconvert")

    videosink = Gst.ElementFactory.make("autovideosink", "videosink")
    if videosink is None:
        raise RuntimeError("缺少 autovideosink")

    for element in (src, decodebin, videoconvert, videosink):
        pipeline.add(element)

    if not videoconvert.link(videosink):
        raise RuntimeError("videoconvert 链接 videosink 失败")

    def on_decode_pad_added(_decode: Gst.Element, pad: Gst.Pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        struct = caps.get_structure(0) if caps and caps.get_size() > 0 else None
        name = struct.get_name() if struct else ""
        if not name.startswith("video/"):
            return

        sink_pad = videoconvert.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        pad.link(sink_pad)

    decodebin.connect("pad-added", on_decode_pad_added)

    def on_rtsp_pad_added(_src: Gst.Element, pad: Gst.Pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        struct = caps.get_structure(0) if caps and caps.get_size() > 0 else None
        if struct is None:
            return

        if struct.get_name() != "application/x-rtp":
            return

        media = struct.get_string("media") or ""
        if media != "video":
            return

        encoding = (struct.get_string("encoding-name") or "").upper()
        if encoding == "H264":
            depay_name = "rtph264depay"
        elif encoding in {"H265", "HEVC"}:
            depay_name = "rtph265depay"
        else:
            print(f"不支持的编码: {encoding}", file=sys.stderr)
            return

        depay = Gst.ElementFactory.make(depay_name, None)
        if depay is None:
            print(f"缺少 {depay_name} 插件", file=sys.stderr)
            return

        queue = Gst.ElementFactory.make("queue", None)
        if queue is None:
            print("缺少 queue 插件", file=sys.stderr)
            return

        pipeline.add(queue)
        pipeline.add(depay)
        queue.sync_state_with_parent()
        depay.sync_state_with_parent()

        if not queue.link(depay):
            print("queue 链接 depay 失败", file=sys.stderr)
            return

        if not depay.link(decodebin):
            print("depay 链接 decodebin 失败", file=sys.stderr)
            return

        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None:
            print("queue 没有 sink pad", file=sys.stderr)
            return

        pad.link(sink_pad)

    src.connect("pad-added", on_rtsp_pad_added)
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_RTSP_URL)
    parser.add_argument("--transport", choices=["tcp", "udp", "udp-mcast", "auto"], default="tcp")
    parser.add_argument("--latency", type=int, default=200)
    parser.add_argument("--user-agent", default="VLC/3.0.0")
    args = parser.parse_args()

    Gst.init(None)
    pipeline = build_pipeline(args.url, transport=args.transport, latency_ms=args.latency, user_agent=args.user_agent)
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus: Gst.Bus, msg: Gst.Message) -> None:
        msg_type = msg.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            if debug:
                print(f"GStreamer 错误: {err} | debug: {debug}", file=sys.stderr)
            else:
                print(f"GStreamer 错误: {err}", file=sys.stderr)
            loop.quit()

        elif msg_type == Gst.MessageType.EOS:
            loop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            if debug:
                print(f"GStreamer 警告: {err} | debug: {debug}", file=sys.stderr)
            else:
                print(f"GStreamer 警告: {err}", file=sys.stderr)

        elif msg_type == Gst.MessageType.STATE_CHANGED and msg.src == pipeline:
            old, new, pending = msg.parse_state_changed()
            print(f"状态: {old.value_nick} -> {new.value_nick} (pending={pending.value_nick})")

    bus.connect("message", on_message)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("进入 PLAYING 状态失败", file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        return 2

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
