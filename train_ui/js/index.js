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
            // for (let i = 0; i < x.length; i++) {
            //     let selected_index = 0
            //     if (i == selected) {
            //         selected_index = 1
            //     }
            //     retval.push([x[i], selected])
            // }
            // return retval;
        }
    }
    return new SDClass();
})();