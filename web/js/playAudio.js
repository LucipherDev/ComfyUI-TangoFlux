import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
  name: "TangoFlux.playAudio",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "TangoFluxVAEDecodeAndPlay") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;

      nodeType.prototype.onNodeCreated = async function () {
        originalNodeCreated?.apply(this, arguments);
        this.widgets_count = this.widgets?.length || 0;

        this.addAudioWidgets = (audios) => {
          if (this.widgets) {
              for (let i = 0; i < this.widgets.length; i++) {
                  if (this.widgets[i].name.startsWith("_playaudio")) {
                      this.widgets[i].onRemove?.();
                  }
              }
              this.widgets.length = this.widgets_count;
          }

          let index = 0
          for (const params of audios) {
              const audioElement = document.createElement("audio");
              audioElement.controls = true;

              this.addDOMWidget("_playaudio" + index, "playaudio", audioElement, {
                serialize: false,
                hideOnZoom: false,
              });
              audioElement.src = api.apiURL(
                `/tangoflux/playaudio?${new URLSearchParams(params)}`
              );
              index++
          }

          requestAnimationFrame(() => {
            const newSize = this.computeSize();
            newSize[0] = Math.max(newSize[0], this.size[0]);
            newSize[1] = Math.max(newSize[1], this.size[1]);
            this.onResize?.(newSize);
            app.graph.setDirtyCanvas(true, false);
          });
        };
      };

      const originalNodeExecuted = nodeType.prototype.onExecuted;

      nodeType.prototype.onExecuted = async function (message) {
        originalNodeExecuted?.apply(this, arguments);
        if (message?.audios) {
          this.addAudioWidgets(message.audios);
        }
      };
    }
  },
});
