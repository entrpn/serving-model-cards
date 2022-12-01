window.SD = (() => {
    console.log("Loaded js");

    class ElementCache{
        constructor () {
            this.root = document.querySelector('gradio-app').shadowRoot;
        }
        get (selector) {
            return this.root.querySelector(selector);
        }
    }

    class SDClass {
        el = new ElementCache()

        getGallerySelectedItem({x, element_id}) {
            if (!Array.isArray(x) || x.length === 0) return;
            let retval = []
            const gallery = this.el.get(`#${element_id}`);
            let selected = gallery.querySelector(`.\\!ring-2`);
            selected = selected ? [...selected.parentNode.children].indexOf(selected) : 0;
            return [[x[selected],selected]]
        }
    }
    return new SDClass();
})();