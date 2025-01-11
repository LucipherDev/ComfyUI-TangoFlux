import os
import server
import folder_paths

web = server.web


@server.PromptServer.instance.routes.get("/tangoflux/playaudio")
async def play_audio(request):
    query = request.rel_url.query

    filename = query.get("filename", None)

    if filename is None:
        return web.Response(status=404)
    
    if filename[0] == "/" or ".." in filename:
        return web.Response(status=403)

    filename, output_dir = folder_paths.annotated_filepath(filename)

    if not output_dir:
        file_type = query.get("type", "output")
        output_dir = folder_paths.get_directory_by_type(file_type)

    if output_dir is None:
        return web.Response(status=400)

    subfolder = query.get("subfolder", None)
    if subfolder:
        full_output_dir = os.path.join(output_dir, subfolder)
        if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
            return web.Response(status=403)
        output_dir = full_output_dir

    filename = os.path.basename(filename)
    file_path = os.path.join(output_dir, filename)

    if not os.path.isfile(file_path):
        return web.Response(status=404)

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    content_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
    }

    content_type = content_types.get(ext, None)

    if content_type is None:
        return web.Response(status=400)

    try:
        with open(file_path, "rb") as file:
            data = file.read()
    except:
        return web.Response(status=500)

    return web.Response(body=data, content_type=content_type)
