import xml
import xml.dom.minidom

class IPC_node:
	def __init__(self, parent, t):
		self.children = []

		self.parent = parent # link child to parent
		if parent:
			parent.add_child(self) # link parent to child

		self.type = t
		self.symbol = None
		self.description = None

	def add_child(self, child):
		self.children.append(child)

	def add_symbol(self, attributes):
		for a in attributes.items():
			if a[0] == 'symbol':
				self.symbol = a[1]


class IPC_tree:
	def __init__(self, xml_tree):
		self.xml_tree = xml_tree
		self.section_count = 0
		self.class_count = 0
		self.subclass_count = 0
		self.sections = []
		self.classes = []
		self.subclasses = []
		self.subclass_nodes = []

		self.root = IPC_node(None, 'root')



	def parse_tree(self):
		node = self.xml_tree.firstChild
		for n in node.childNodes:
			if n.nodeName == 'ipcEntry' and ('kind', 's') in n.attributes.items():
				s = IPC_node(self.root, 'section')
				s.add_symbol(n.attributes)
				self.section_count += 1
				self.sections.append(s.symbol)
				self.parse_section(n, s)


	def parse_section(self, section_node, section):
		for n in section_node.childNodes:
			if n.nodeName == 'ipcEntry' and ('kind', 'c') in n.attributes.items():
				c = IPC_node(section, 'class')
				c.add_symbol(n.attributes)

				self.class_count += 1
				self.classes.append(c.symbol)

				self.parse_class(n, c)



	def parse_class(self, class_node, clas):
		for n in class_node.childNodes:
			if n.nodeName == 'ipcEntry' and ('kind', 'u') in n.attributes.items():
				sub_c = IPC_node(clas, 'subclass')
				sub_c.add_symbol(n.attributes)

				#clas.add_child(sub_c)
				self.subclass_count += 1
				self.subclasses.append(sub_c.symbol)
				
				self.subclass_nodes.append(n)

				self.parse_subclass(n, sub_c)


	def parse_subclass(self, sub_class_node, sub_class):
		'''Extract in text descriptions for subclass'''
		pass



xml_tree = xml.dom.minidom.parse('../data/EN_ipc_scheme_20190101.xml')

IPC = IPC_tree(xml_tree)
IPC.parse_tree()

print('There are a total of', IPC.section_count, 'sections')
print('                    ', IPC.class_count, 'classes')
print('                    ', IPC.subclass_count, 'subclasses.')

#print(IPC.subclasses)



