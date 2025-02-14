package("myopencv")

    on_load(function (package)
        package:set("installdir", path.join(os.scriptdir(), "windows/x64/release"))
    end)


    on_fetch(function (package)
        local result = {}
        result.linkdirs = package:installdir("lib")
        package:addenv("PATH", package:installdir("bin"))
        result.includedirs = package:installdir("include")
        result.links = {"ade",
                        "libjpeg-turbo",
                        "libpng",
                        "libprotobuf",
                        "libtiff",
                        "libwebp",
                        "opencv_aruco453",
                        "opencv_barcode453",
                        "opencv_bgsegm453",
                        "opencv_bioinspired453",
                        "opencv_calib3d453",
                        "opencv_ccalib453",
                        "opencv_core453",
                        "opencv_datasets453",
                        "opencv_dnn453",
                        "opencv_dnn_objdetect453",
                        "opencv_dnn_superres453",
                        "opencv_dpm453",
                        "opencv_face453",
                        "opencv_features2d453",
                        "opencv_flann453",
                        "opencv_fuzzy453",
                        "opencv_gapi453",
                        "opencv_hfs453",
                        "opencv_highgui453",
                        "opencv_imgcodecs453",
                        "opencv_imgproc453",
                        "opencv_img_hash453",
                        "opencv_intensity_transform453",
                        "opencv_line_descriptor453",
                        "opencv_mcc453",
                        "opencv_ml453",
                        "opencv_objdetect453",
                        "opencv_optflow453",
                        "opencv_phase_unwrapping453",
                        "opencv_photo453",
                        "opencv_plot453",
                        "opencv_quality453",
                        "opencv_rapid453",
                        "opencv_reg453",
                        "opencv_rgbd453",
                        "opencv_saliency453",
                        "opencv_shape453",
                        "opencv_stereo453",
                        "opencv_stitching453",
                        "opencv_structured_light453",
                        "opencv_superres453",
                        "opencv_surface_matching453",
                        "opencv_text453",
                        "opencv_tracking453",
                        "opencv_video453",
                        "opencv_videoio453",
                        "opencv_videostab453",
                        "opencv_wechat_qrcode453",
                        "opencv_xfeatures2d453",
                        "opencv_ximgproc453",
                        "opencv_xobjdetect453",
                        "opencv_xphoto453",
                        "quirc",
                        "zlib"}
        return result
    end)

    on_test(function (package)
        print("Package: %s", package:installdir())
        print("Version: %s", package:linkdirs())
        print("Description: %s", package:includedirs())
    end)





